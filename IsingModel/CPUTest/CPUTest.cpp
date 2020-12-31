#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <new>
#include <_Time.h>
#include <_Bit.h>
#ifdef _WIN32
#include <intrin.h>
#include <Windows.h>
#include <process.h>
#define popc(x) __popcnt(x)
#define ffs(x) _tzcnt_u32(x)
#else
// #include <immintrin.h>
#define popc(x) __builtin_popcount(x)
#endif
//#define PRINTDEBUGINFO

//dimension: 1 + 1
//size: N = 128
//use int32 instead of int64 because CUDA's support for int64 is simulated


constexpr unsigned int powd(unsigned int a, unsigned int n)
{
	if (n)return powd(a, n - 1) * a;
	else return 1;
}

template<unsigned int N, unsigned int SpaceDim = 1>
struct Grid
{
	static constexpr unsigned int NMinus1 = N - 1;
	static constexpr unsigned int BrickNum = N / 32;
	static constexpr unsigned int BrickMinus1 = BrickNum - 1;
	static constexpr unsigned int BrickNumTotal = BrickNum * powd(N, SpaceDim);
	static constexpr unsigned int SpinNum = powd(N, SpaceDim + 1);
	static constexpr unsigned int GridSize = powd(N, SpaceDim) * BrickNum * sizeof(unsigned int);
	static constexpr unsigned int GridUSize = powd(N, SpaceDim) * sizeof(unsigned int);
	static constexpr unsigned int GridKinksSize = powd(N, SpaceDim) * SpaceDim * BrickNum * sizeof(unsigned int);
	static constexpr unsigned int GridKinksNumSize = powd(N, SpaceDim) * SpaceDim * sizeof(unsigned int);

	static constexpr float Pa = 1.f / 3;
	static constexpr float Pb = 1.f / 3;
	static constexpr float Pc = 1.f / 6;
	static constexpr float APa = Pa;
	static constexpr float APb = Pa + Pb;
	static constexpr float APc0 = Pa + Pb + Pc;


	/*
	static constexpr float h = 0.04f;
		static constexpr unsigned int tauB = tauA;			//\tau_b
		static constexpr unsigned int tauC = tauA;			//\tau_c
		static constexpr unsigned int tauAD2 = tauA / 2;	//\dfrac{\tau_a}{2}
		static constexpr unsigned int tauBD2 = tauB / 2;	//\dfrac{\tau_b}{2}
		static constexpr unsigned int tauCD2 = tauC / 2;	//\dfrac{\tau_c}{2}
		static constexpr unsigned int tauBM1 = tauB - 1;	//\tau_b-1
		static constexpr unsigned int tauCM1 = tauC - 1;	//\tau_c-1
		static constexpr unsigned int tauBD2M1 = tauBD2 - 1;//\dfrac{\tau_b}{2}-1
		static constexpr unsigned int tauCD2M1 = tauCD2 - 1;//\dfrac{\tau_c}{2}-1
		static constexpr float tauCInv = 1.f / tauC;
		static constexpr float PaTauA = Pa * tauA;
		static constexpr float PaTauAInv = 1 / PaTauA;*/

	float h;

	unsigned int tauA;		//\tau_a
	unsigned int tauB;		//\tau_b
	unsigned int tauC;		//\tau_c
	unsigned int tauAD2;	//\dfrac{\tau_a}{2}
	unsigned int tauBD2;	//\dfrac{\tau_b}{2}
	unsigned int tauCD2;	//\dfrac{\tau_c}{2}
	unsigned int tauBM1;	//\tau_b-1
	unsigned int tauCM1;	//\tau_c-1
	unsigned int tauBD2M1;	//\dfrac{\tau_b}{2}-1
	unsigned int tauCD2M1;	//\dfrac{\tau_c}{2}-1
	float tauCInv = 1.f / tauC;
	float PaTauA = Pa * tauA;
	float PaTauAInv = 1 / PaTauA;

	std::uniform_real_distribution<float> rd;
	std::uniform_int_distribution<unsigned int> rdint;
	std::uniform_int_distribution<unsigned int> rdWorldLineDim;
	std::uniform_int_distribution<int> rdWorldLineDir;
	std::uniform_int_distribution<unsigned int> rdtauA;
	//if return non-negetive, then plus 1 to make sure the result is in [-tauBD2, tauBD2]\{0}
	std::uniform_int_distribution<int> rdDeltaB;
	//if return non-negetive, then plus 1 to make sure the result is in [-tauCD2, tauCD2]\{0}
	std::uniform_int_distribution<int> rdDeltaC;
	std::uniform_int_distribution<int> rdDeltaC1;

	std::mt19937 mt;
	Timer timer;

	enum class OperationType
	{
		None,
		Create,
		Annihilate,
		Move,
		Insert,
		Delete
	};

	//use pointer instead of multi-dimension array to simulate CUDA
	unsigned int* grid;
	//stores U
	unsigned int* gridU;
	//stores kink, gridKinks[z][y][x][dim][brick]=gridKinks[brick + BrickNum * (dim + SpaceDim * (x + N * (y + N * z)))],
	//a kink of t = i is something between t = i and t = i + 1
	unsigned int* gridKinks;

	//DFS traverse of the grid
	unsigned int* gridTraverse;

	int Ti, Tm;//-1 means not created, does not need modulo
	int Xi, Xm;

	OperationType operation;

	unsigned int Rm;
	unsigned int rounds;
	unsigned int steps;
	//different direction: 1
	//same      direction: 0
	//0: n = 1
	unsigned long long correlation[N / 2];
	unsigned long long averageSpin;

	unsigned int tp0[BrickNum];
	unsigned int tp1[BrickNum];

	int deltaMove;
	int deltaInsert;
	int deltaDelete;

#define get(x, y) (((x) >> (y)) & 1)
#define set(x, y) ((x) ^= (1u << (y)))
#define setTo1(x, y) ((x) |= (1u << (y)))
#define getGrid(origin, x) (get((origin)[(x)>>5], (x) & 31))
#define setGrid(origin, x) (set((origin)[(x)>>5], (x) & 31))
#define setGridTo1(origin, x) (setTo1((origin)[(x)>>5], (x) & 31))

	Grid(unsigned int seed)
		:
		rd(0, 1),
		rdint(0, N - 1),
		rdWorldLineDim(0, SpaceDim - 1),
		rdWorldLineDir(0, 1),
		mt(time(nullptr) + seed),

		grid((unsigned int*)malloc(GridSize)),
		gridU((unsigned int*)malloc(GridUSize)),
		gridKinks((unsigned int*)malloc(GridKinksSize)),
		gridTraverse((unsigned int*)malloc(GridSize)),
		operation(OperationType::None)
	{
	}
	~Grid()
	{
		free(grid);
		free(gridU);
		free(gridKinks);
		free(gridTraverse);
		//free(gridKinksTraverse);
	}
	//Clear the grid to original state
	void clear()
	{
		memset(grid, 0, GridSize);
		memset(gridU, 0, GridUSize);
		memset(gridKinks, 0, GridKinksSize);
		memset(correlation, 0, sizeof(correlation));
		Ti = -1;
		Rm = 0;
		rounds = 0;
		steps = 0;
		averageSpin = 0;
		deltaMove = 0;
		deltaInsert = 0;
		deltaDelete = 0;
	}
	//initialize parameters according to h and then clear
	void init(float _h)
	{
		h = _h;
#define IF(x) (x)?
#define THEN(x) (x):
#define ELSE(x) (x)
		tauA = 4;
		//IF(1.f / h < 4)
		//THEN(4)
		//ELSE(
		//	IF(1.f / h > N / 2)
		//	THEN(N / 2)
		//	ELSE((unsigned int)(1.f / h) & (-2)));
#undef IF
#undef THEN
#undef ELSE
		tauB = tauA;
		tauC = tauA;
		tauAD2 = tauA / 2;
		tauBD2 = tauB / 2;
		tauCD2 = tauC / 2;
		tauBM1 = tauB - 1;
		tauCM1 = tauC - 1;
		tauBD2M1 = tauBD2 - 1;
		tauCD2M1 = tauCD2 - 1;
		tauCInv = 1.f / tauC;
		PaTauA = Pa * tauA;
		PaTauAInv = 1 / PaTauA;

		rdtauA = std::uniform_int_distribution<unsigned int>(1, tauA);
		rdDeltaB = std::uniform_int_distribution<int>(-int(tauBD2), tauBD2 - 1);
		rdDeltaC = std::uniform_int_distribution<int>(-int(tauCD2), tauCD2 - 1);
		rdDeltaC1 = std::uniform_int_distribution<int>(-int(tauCD2), tauCD2);

		clear();
	}
	inline unsigned int* gridOrigin(unsigned int X)
	{
		return grid + BrickNum * X;
	}
	inline unsigned int* gridKinksOrigin(unsigned int X, unsigned int dim)
	{
		return gridKinks + BrickNum * (dim + SpaceDim * X);
	}
	inline unsigned int* gridOrigin(unsigned int* g, unsigned int X)
	{
		return g + BrickNum * X;
	}
	inline unsigned int* gridKinksOrigin(unsigned int* g, unsigned int X, unsigned int dim)
	{
		return g + BrickNum * (dim + SpaceDim * X);
	}
	//logic or on section [t0, t1] in different lins of gridKinks, stores in tp0
	void kinksLogicOr(int Xk, int Xk1, int Xk2, unsigned int t0, unsigned int t1)
	{
		unsigned int* origin(gridKinksOrigin(Xk, 0));
		unsigned int* origin1(gridKinksOrigin(Xk1, 0));
		unsigned int* origin2(gridKinksOrigin(Xk2, 0));
		//for (unsigned int c0(0); c0 < BrickNum; ++c0)
		//	tp0[c0] = origin[c0] | origin1[c0] | origin2[c0];
		unsigned int Ua(t0 >> 5);
		unsigned int Ub(t1 >> 5);
		if (t0 <= t1)
		{
			for (unsigned int c0(Ua); c0 <= Ub; ++c0)
				tp0[c0] = origin[c0] | origin1[c0] | origin2[c0];
		}
		else
		{
			for (unsigned int c0(Ua); c0 <= (BrickNum + Ub); ++c0)
			{
				unsigned int p(c0 & BrickMinus1);
				tp0[p] = origin[p] | origin1[p] | origin2[p];
			}
		}
	}
	//Flips spins in [t0, t1];
	//make sure that 0 <= t0 < N, 0 <= t1 < N;
	//stores the result in tp, return the original U of the flipped spins
	unsigned int flipSpins(unsigned int* tp, unsigned int X, unsigned int t0, unsigned int t1)
	{
		unsigned int* origin(gridOrigin(X));
		unsigned int Ua(t0 >> 5);
		unsigned int Ub(t1 >> 5);
		unsigned int Da(t0 & 31);
		unsigned int Db(t1 & 31);
		unsigned int U(0);
		if (t0 <= t1)
		{
			if (Ua == Ub)
			{
				unsigned int mask(0xffffffff);
				mask >>= (31 - Db + Da);
				mask <<= Da;
				U += popc(origin[Ua] & mask);
				tp[Ua] = origin[Ua] ^ mask;
			}
			else
			{
				unsigned int mask(0xffffffff);
				mask <<= Da;
				U += popc(origin[Ua] & mask);
				tp[Ua] = origin[Ua] ^ mask;
				for (unsigned int c0(Ua + 1); c0 < Ub; ++c0)
				{
					U += popc(origin[c0]);
					tp[c0] = origin[c0] ^ 0xffffffff;
				}
				mask = 0xffffffff;
				mask >>= (31 - Db);
				U += popc(origin[Ub] & mask);
				tp[Ub] = origin[Ub] ^ mask;
			}
		}
		else
		{
			if (Ua == Ub)
			{
				unsigned int mask(0xffffffff);
				mask >>= (33 - Da + Db);
				mask <<= (Db + 1);
				mask = ~mask;
				U += popc(origin[Ua] & mask);
				tp[Ua] = origin[Ua] ^ mask;
			}
			else
			{
				unsigned int mask(0xffffffff);
				mask <<= Da;
				U += popc(origin[Ua] & mask);
				tp[Ua] = origin[Ua] ^ mask;

				mask = 0xffffffff;
				mask >>= (31 - Db);
				U += popc(origin[Ub] & mask);
				tp[Ub] = origin[Ub] ^ mask;
			}
			for (unsigned int c0(Ua + 1); c0 < BrickNum + Ub; ++c0)
			{
				unsigned int p(c0 & BrickMinus1);
				U += popc(origin[p]);
				tp[p] = origin[p] ^ 0xffffffff;
			}
		}
		return U;
	}
	//copy the states changed by flipSpins
	//parameters should be the same as flipSpins
	void copySpins(unsigned int* tp, unsigned int X, unsigned int t0, unsigned int t1)
	{
		unsigned int* origin(gridOrigin(X));
		unsigned int Ua(t0 >> 5);
		unsigned int Ub(t1 >> 5);
		if (t0 <= t1)
		{
			origin[Ua] = tp[Ua];
			if (Ua != Ub)
			{
				for (unsigned int c0(Ua + 1); c0 < Ub; ++c0)
					origin[c0] = tp[c0];
				origin[Ub] = tp[Ub];
			}
		}
		else
		{
			origin[Ua] = tp[Ua];
			if (Ua != Ub)origin[Ub] = tp[Ub];
			for (unsigned int c0(Ua + 1); c0 < BrickNum + Ub; ++c0)
			{
				unsigned int p(c0 & BrickMinus1);
				origin[p] = tp[p];
			}
		}
	}
	//calculate the sum of all spins along with X
	unsigned int spinUpNum(unsigned int X)
	{
		unsigned int a(0);
		unsigned int* origin(gridOrigin(X));
		for (unsigned int c0(0); c0 < BrickNum; ++c0)
			a += popc(origin[c0]);
		return a;
	}
	//total number of spin-ups
	void spinUpNum()
	{
		for (unsigned int c0(0); c0 < BrickNumTotal; ++c0)
			averageSpin += popc(grid[c0]);
	}
	//calculate the kinks number of world line origin in [t0, t1]
	unsigned int kinksNum(unsigned int* origin, unsigned int t0, unsigned int t1)
	{
		unsigned int Ua(t0 >> 5);
		unsigned int Ub(t1 >> 5);
		unsigned int Da(t0 & 31);
		unsigned int Db(t1 & 31);
		unsigned int kinksNum(0);
		if (t0 <= t1)
		{
			if (Ua == Ub)
			{
				unsigned int mask(0xffffffff);
				mask >>= (31 - Db + Da);
				mask <<= Da;
				kinksNum += popc(origin[Ua] & mask);
			}
			else
			{
				unsigned int mask(0xffffffff);
				mask <<= Da;
				kinksNum += popc(origin[Ua] & mask);
				for (unsigned int c0(Ua + 1); c0 < Ub; ++c0)
					kinksNum += popc(origin[c0]);
				mask = 0xffffffff;
				mask >>= (31 - Db);
				kinksNum += popc(origin[Ub] & mask);
			}
		}
		else
		{
			if (Ua == Ub)
			{
				unsigned int mask(0xffffffff);
				mask >>= (33 - Da + Db);
				mask <<= (Db + 1);
				mask = ~mask;
				kinksNum += popc(origin[Ua] & mask);
			}
			else
			{
				unsigned int mask(0xffffffff);
				mask <<= Da;
				kinksNum += popc(origin[Ua] & mask);
				mask = 0xffffffff;
				mask >>= (31 - Db);
				kinksNum += popc(origin[Ub] & mask);
			}
			for (unsigned int c0(Ua + 1); c0 < BrickNum + Ub; ++c0)
				kinksNum += popc(origin[c0 & BrickMinus1]);
		}
		return kinksNum;
	}
	//create defectes
	void createDefects()
	{
#ifdef PRINTDEBUGINFO
		printf("Create\t\t");
#endif
		Xi = Xm = rdint(mt);
		Ti = rdint(mt);
		int dtauM(rdtauA(mt));
		Tm = (Ti + dtauM) & NMinus1;
		//Flip spins in (Ti, Tm]
		int dU(flipSpins(tp0, Xi, (Ti + 1) & NMinus1, Tm));
		dtauM -= 2 * dU;//variation of U
		float acceptance(PaTauA * expf(h * dtauM));
		if (rd(mt) < acceptance)
		{
			copySpins(tp0, Xi, (Ti + 1) & NMinus1, Tm);
			gridU[Xi] += dtauM;
			operation = OperationType::Create;
#ifdef PRINTDEBUGINFO
			printf("accepted\t");
			printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u\n", Xi, Xm, Ti, Tm);
#endif
		}
		else
		{
			Ti = -1;
#ifdef PRINTDEBUGINFO
			printf("denied\n");
#endif
		}
	}
	//annihilate defectes
	void annihilateDefects()
	{
#ifdef PRINTDEBUGINFO
		printf("Annihilate\t");
#endif
		bool flag(false);
		unsigned int dt(abs(Ti - Tm));
		int dtauM(dt <= N - dt ? dt : N - dt);
		if (dtauM <= tauAD2)
		{
			float acceptance(PaTauAInv);
			unsigned int t0, t1;
			if (dt)
			{
				if ((Ti > Tm) ^ (dt <= tauAD2))  t0 = Ti, t1 = Tm;
				else t0 = Tm, t1 = Ti;
				int dU(flipSpins(tp0, Xi & NMinus1, (t0 + 1) & NMinus1, t1));
				dtauM -= 2 * dU;//variation of U
				acceptance *= expf(h * dtauM);
			}
			if (rd(mt) < acceptance)
			{
				if (dt)
				{
					copySpins(tp0, Xi & NMinus1, (t0 + 1) & NMinus1, t1);
					gridU[Xi & NMinus1] += dtauM;
				}
				Ti = -1;
				rounds++;
				operation = OperationType::Annihilate;
#ifdef PRINTDEBUGINFO
				printf("accepted\t");
				printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u\n", Xi, Xm, Ti, Tm);
#endif
				flag = true;
			}
		}
#ifdef PRINTDEBUGINFO
		if (!flag)printf("denied\n");
#endif
	}
	//move Tm
	void moveMT()
	{
#ifdef PRINTDEBUGINFO
		printf("Move\t\t");
#endif
		bool flag(false);
		unsigned int Tn;
		unsigned int t0, t1;
		int dd(rdDeltaB(mt)), dd0;
		if (dd >= 0)
		{
			dd++;
			Tn = (Tm + dd) & NMinus1;
			t0 = Tm, t1 = Tn;
		}
		else
		{
			Tn = (Tm + dd) & NMinus1;
			t0 = Tn, t1 = Tm;
		}
		dd0 = abs(dd);
		int dU(flipSpins(tp0, Xm & NMinus1, (t0 + 1) & NMinus1, t1));
		dd0 -= 2 * dU;//variation of U
		float acceptance(expf(h * dd0));
		if (rd(mt) < acceptance)
		{
			copySpins(tp0, Xm & NMinus1, (t0 + 1) & NMinus1, t1);
			gridU[Xm & NMinus1] += dd0;
			Tm = Tn;
			deltaMove += dd;
			operation = OperationType::Move;
#ifdef PRINTDEBUGINFO
			printf("accepted\t");
			printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u\n", Xi, Xm, Ti, Tm);
#endif
			flag = true;
		}
#ifdef PRINTDEBUGINFO
		if (!flag)printf("denied\n");
#endif
	}
	//insert a kink
	void insertKink()
	{
#ifdef PRINTDEBUGINFO
		printf("Insert\t\t");
		bool flag(false);
#endif
		int dim(rdWorldLineDim(mt));//which dimension that the movement takes 
		int dir(rdWorldLineDir(mt));//direction of the movement, 0 means -1, 1 means 1
		int Xn(Xm + (2 * dir - 1));//neighbour of Xm, new Xm
		int Xk((dir ? Xm : Xn) & NMinus1);//choose the index of gridKinks
		unsigned int Tn;//position of the add kink
		unsigned int* origin(gridKinksOrigin(Xk, dim));
		unsigned int t0, t1;
		int dd, dd0, dd1;
		//kinks number in [Tm - tauCD2, Tm + tauCD2 - 1)
		unsigned int kn(kinksNum(origin, (Tm - tauCD2) & NMinus1, (Tm + tauCD2M1) & NMinus1));
		//make sure that there is space for a new kink
		if (tauCM1 != kn - getGrid(origin, Tm))
			do
			{
				//choose a position which has no kink
				dd = rdDeltaC(mt);
				if (dd >= 0)
				{
					dd++;
					Tn = (Tm + dd) & NMinus1;
					t0 = (Tm + 1) & NMinus1, t1 = Tn;
				}
				else
				{
					Tn = (Tm + dd) & NMinus1;
					t0 = (Tn + 1) & NMinus1, t1 = Tm;
				}
			} while (getGrid(origin, Tn));
		else
		{
#ifdef PRINTDEBUGINFO
			printf("denied\n");
#endif
			return;
		}
		dd1 = dd0 = abs(dd);
		int dU0(flipSpins(tp0, Xm & NMinus1, t0, t1));
		dd0 -= 2 * dU0;//variation of U
		int dU1(flipSpins(tp1, Xn & NMinus1, t0, t1));
		dd1 -= 2 * dU1;//variation of U
		float acceptance((tauC * expf(h * (dd0 + dd1)) / (kn + 1)));
		if (rd(mt) < acceptance)
		{
			copySpins(tp0, Xm & NMinus1, t0, t1);
			copySpins(tp1, Xn & NMinus1, t0, t1);
			setGrid(origin, Tn);
			gridU[Xm & NMinus1] += dd0;
			gridU[Xn & NMinus1] += dd1;
			Xm = Xn;
			operation = OperationType::Insert;
#ifdef PRINTDEBUGINFO
			printf("accepted\t");
			printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u, Tn: %4u\n", Xi, Xm, Ti, Tm, Tn);
			flag = true;
#endif
		}
#ifdef PRINTDEBUGINFO
		if (!flag)printf("denied\n");
#endif
	}
	//insert a kink (two neighbour kinks cannot have the same t)
	void insertKinkLimited()
	{
#ifdef PRINTDEBUGINFO
		printf("InsertLimited\t");
		bool flag(false);
#endif
		int dim(rdWorldLineDim(mt));//which dimension that the movement takes 
		int dir(rdWorldLineDir(mt));//direction of the movement, 0 means -1, 1 means 1
		int Xn(Xm + (2 * dir - 1));//neighbour of Xm, new Xm
		int Xk, Xk1, Xk2;
		if (dir)
		{
			Xk = Xm & NMinus1;
			Xk1 = Xn & NMinus1;
			Xk2 = (Xm + NMinus1) & NMinus1;
		}
		else
		{
			Xk = Xn & NMinus1;
			Xk1 = Xm & NMinus1;
			Xk2 = (Xn + NMinus1) & NMinus1;
		}
		unsigned int Tn;//position of the add kink
		unsigned int tb((Tm - tauCD2) & NMinus1), te((Tm + tauCD2) & NMinus1);
		kinksLogicOr(Xk, Xk1, Xk2, tb, te);
		unsigned int t0, t1;
		int dd, dd0, dd1;
		unsigned int* origin(gridKinksOrigin(Xk, dim));
		unsigned int kn(kinksNum(tp0, tb, te));
		//kinks number in [Tm - tauCD2, Tm + tauCD2 - 1)
		unsigned int nk(kinksNum(origin, tb, te));
		//make sure that there is space for a new kink
		if (tauC != kn - getGrid(tp0, Tm))
			do
			{
				//choose a position which has no kink
				dd = rdDeltaC(mt);
				if (dd >= 0)
				{
					dd++;
					Tn = (Tm + dd) & NMinus1;
					t0 = (Tm + 1) & NMinus1, t1 = Tn;
				}
				else
				{
					Tn = (Tm + dd) & NMinus1;
					t0 = (Tn + 1) & NMinus1, t1 = Tm;
				}
			} while (getGrid(tp0, Tn));
		else
		{
#ifdef PRINTDEBUGINFO
			printf("denied\n");
#endif
			return;
		}
		dd1 = dd0 = abs(dd);
		int dU0(flipSpins(tp0, Xm & NMinus1, t0, t1));
		dd0 -= 2 * dU0;//variation of U
		int dU1(flipSpins(tp1, Xn & NMinus1, t0, t1));
		dd1 -= 2 * dU1;//variation of U
		//still nk because the delete operation doesn't check the other two kink lines
		float acceptance((tauC * expf(h * (dd0 + dd1)) / (nk + 1)));
		if (rd(mt) < acceptance)
		{
			copySpins(tp0, Xm & NMinus1, t0, t1);
			copySpins(tp1, Xn & NMinus1, t0, t1);
			setGrid(origin, Tn);
			gridU[Xm & NMinus1] += dd0;
			gridU[Xn & NMinus1] += dd1;
			Xm = Xn;
			deltaMove += dd;
			operation = OperationType::Insert;
#ifdef PRINTDEBUGINFO
			printf("accepted\t");
			printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u, Tn: %4u\n", Xi, Xm, Ti, Tm, Tn);
			flag = true;
#endif
		}
#ifdef PRINTDEBUGINFO
		if (!flag)printf("denied\n");
#endif
	}
	//delete a kink
	void deleteKink()
	{
#ifdef PRINTDEBUGINFO
		printf("Delete\t\t");
		bool flag(false);
#endif
		int dim(rdWorldLineDim(mt));//which dimension that the movement takes 
		int dir(rdWorldLineDir(mt));//direction of the movement, 0 means -1, 1 means 1
		int Xn(Xm + (2 * dir - 1));//neighbour of Xm, new Xm
		int Xk((dir ? Xm : Xn) & NMinus1);//choose the index of gridKinks
		unsigned int Tn;//position of the add kink
		unsigned int* origin(gridKinksOrigin(Xk, dim));
		unsigned int t0, t1;
		int dd, dd0, dd1;
		//kinks number in [Tm - tauCD2, Tm + tauCD2 - 1)
		unsigned int kn(kinksNum(origin, (Tm - tauCD2) & NMinus1, (Tm + tauCD2) & NMinus1));
		//make sure that n_k > 0
		if (kn)
			do
			{
				//choose a position which has a kink
				dd = rdDeltaC1(mt);
				Tn = (Tm + dd) & NMinus1;
				if (dd >= 0)t0 = Tm, t1 = Tn;
				else t0 = Tn, t1 = Tm;
			} while (!getGrid(origin, Tn));
		else
		{
#ifdef PRINTDEBUGINFO
			printf("denied\n");
#endif
			return;
		}
		float acceptance(kn * tauCInv);
		if (dd)
		{
			dd1 = dd0 = abs(dd);
			int dU0(flipSpins(tp0, Xm & NMinus1, (t0 + 1) & NMinus1, t1));
			dd0 -= 2 * dU0;//variation of U
			int dU1(flipSpins(tp1, Xn & NMinus1, (t0 + 1) & NMinus1, t1));
			dd1 -= 2 * dU1;//variation of U
			acceptance *= expf(h * (dd0 + dd1));
		}
		if (rd(mt) < acceptance)
		{
			if (dd)
			{
				copySpins(tp0, Xm & NMinus1, (t0 + 1) & NMinus1, t1);
				copySpins(tp1, Xn & NMinus1, (t0 + 1) & NMinus1, t1);
				gridU[Xm & NMinus1] += dd0;
				gridU[Xn & NMinus1] += dd1;
			}
			setGrid(origin, Tn);
			Xm = Xn;
			deltaDelete += dd;
			operation = OperationType::Delete;
#ifdef PRINTDEBUGINFO
			printf("accepted\t");
			printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u, Tn: %4u\n", Xi, Xm, Ti, Tm, Tn);
			flag = true;
#endif
		}
#ifdef PRINTDEBUGINFO
		if (!flag)printf("denied\n");
#endif
	}
	//one step of operation
	//TIME consumption (create a loop on average): 64: 1-2 ms
	void operate()
	{
		steps++;
		operation = OperationType::None;
		if (Ti < 0)
		{
			createDefects();
		}
		else
		{
			float r(rd(mt));
			if (r <= APa && !((Xi - Xm) & NMinus1))
			{
				annihilateDefects();
			}
			else if (r <= APb)
			{
				moveMT();
			}
			else if (r <= APc0)
			{
				insertKinkLimited();
			}
			else
			{
				deleteKink();
			}
		}
	}
	//create and annihilate one loop
	void oneLoop()
	{
		do
		{
			operate();
		} while (operation != OperationType::Annihilate);
		Rm += (traverse() != 0);
		calCorrelation();
		spinUpNum();
	}
	//reach balance
	void reachBalance(unsigned int _n)
	{
#ifdef PRINTDEBUGINFO
		printSimInfo();
#endif
		for (unsigned int c0(0); c0 < _n; ++c0)
		{
			do
			{
				operate();
			} while (operation != OperationType::Annihilate);
		}
#ifdef PRINTDEBUGINFO
		printf("Reach Balance: ");
		printResults();
#endif
		Rm = 0;
		rounds = 0;
		steps = 0;
		operation = OperationType::None;
	}
	//some test set
	void simpleTestSet()
	{
		operation = OperationType::Annihilate;
		unsigned int* origin(gridOrigin(0));
		setGrid(origin, 1);
		origin = gridOrigin(1);
		setGrid(origin, 1);

		unsigned int* kinkOrigin(gridKinksOrigin(0, 0));
		setGrid(kinkOrigin, 0);
		setGrid(kinkOrigin, 1);
	}
	//wind number
	unsigned int singleWindingNumber()
	{
		return abs(Xi - Xm) / N;
	}
	//check if the grid has connected kinks
	bool checkConnectedKinks()
	{
		for (unsigned int c0(0); c0 < N; ++c0)
		{
			unsigned int* origin(gridKinks + c0 * BrickNum);
			unsigned int* origin1(gridKinks + ((c0 + 1) & NMinus1) * BrickNum);
			for (unsigned int c1(0); c1 < BrickNum; ++c1)
				if (origin[c1] & origin1[c1])
				{
					printf("Error at x: %d brick: %d!\n", c0, c1);
					return true;
				}
		}
		return false;
	}
	//traverse a grid and return the winding number of the whole grid
	//TIME consumption: 64: 10-20 us
	unsigned int traverse()
	{
#define notPassed(o, oT, x) (getGrid(o, x) ^ getGrid(oT, x))
		unsigned int wdnm(0);
		memset(gridTraverse, 0, GridSize);
		int x(0), t(0);
		int startingBrick(0);
		unsigned int* origin;
		unsigned int* originT;
		unsigned int* kinkOrigin;
		auto setOrigins([this, &origin, &originT, &kinkOrigin](int x)
		{
			int xx(x & NMinus1);
			origin = gridOrigin(xx & NMinus1);
			originT = gridOrigin(gridTraverse, xx);
			kinkOrigin = gridKinksOrigin(xx, 0);
		});
		for (;;)
		{
			setOrigins(x);
			//can start a new loop
			if (notPassed(origin, originT, t))
			{
				setGrid(originT, t);
				int x1(x), t1(t);
				//never goes way back
				int dir(0);// , kinkDir(0);
				do
				{
					unsigned int* origin_tp, * originT_tp;
					//printf("t:%3u\tx:%3u\tdir:%3d\n", t1 & NMinus1, x1 & NMinus1, dir);
					//choose a direction
					int t2((t1 + 1) & NMinus1);
					int x2(x1 + 1);
					// 1.
					if (getGrid(origin, t2) && dir != 6)
					{
						t1 = t2;
						setGridTo1(originT, t1);
						dir = 1;
						continue;
					}
					// a.
					if (getGrid(kinkOrigin, t1))
					{
						origin_tp = gridOrigin(x2 & NMinus1);
						originT_tp = gridOrigin(gridTraverse, x2 & NMinus1);
						bool flag(false);
						// 2.
						if (getGrid(origin_tp, t2) && dir != 7)
						{
							flag = true;
							t1 = t2;
							dir = 2;
						}
						// 3.
						else if (getGrid(origin_tp, t1) && dir != 9)
						{
							flag = true;
							dir = 3;
						}
						if (flag)
						{
							x1 = x2;
							setGridTo1(originT_tp, t1);
							origin = origin_tp;
							originT = gridOrigin(gridTraverse, x1 & NMinus1);
							kinkOrigin = gridKinksOrigin(x1 & NMinus1, 0);
							continue;
						}
					}
					t2 = (t1 + NMinus1) & NMinus1;
					// b.
					if (getGrid(kinkOrigin, t2))
					{
						origin_tp = gridOrigin(x2 & NMinus1);
						originT_tp = gridOrigin(gridTraverse, x2 & NMinus1);
						bool flag(false);
						// 4.
						if (getGrid(origin_tp, t1) && dir != 8)
						{
							flag = true;
							dir = 4;
						}
						// 5.
						else if (getGrid(origin_tp, t2) && dir != 10)
						{
							flag = true;
							t1 = t2;
							dir = 5;
						}
						if (flag)
						{
							x1 = x2;
							setGridTo1(originT_tp, t1);
							origin = origin_tp;
							originT = gridOrigin(gridTraverse, x1 & NMinus1);
							kinkOrigin = gridKinksOrigin(x1 & NMinus1, 0);
							continue;
						}
					}
					// 6.
					if (getGrid(origin, t2) && dir != 1)
					{
						t1 = t2;
						setGridTo1(originT, t1);
						dir = 6;
						continue;
					}
					x2 = x1 - 1;
					unsigned int* kinkOrigin1(gridKinksOrigin(x2 & NMinus1, 0));
					// c.
					if (getGrid(kinkOrigin1, t2))
					{
						origin_tp = gridOrigin(x2 & NMinus1);
						originT_tp = gridOrigin(gridTraverse, x2 & NMinus1);
						bool flag(false);
						// 7.
						if (getGrid(origin_tp, t2) && dir != 2)
						{
							flag = true;
							t1 = t2;
							dir = 7;
						}
						// 8.
						else if (getGrid(origin_tp, t1) && dir != 4)
						{
							flag = true;
							dir = 8;
						}
						if (flag)
						{
							x1 = x2;
							setGridTo1(originT_tp, t1);
							origin = origin_tp;
							originT = gridOrigin(gridTraverse, x1 & NMinus1);
							kinkOrigin = gridKinksOrigin(x1 & NMinus1, 0);
							continue;
						}
					}
					t2 = (t1 + 1) & NMinus1;
					// d.
					if (getGrid(kinkOrigin1, t1))
					{
						origin_tp = gridOrigin(x2 & NMinus1);
						originT_tp = gridOrigin(gridTraverse, x2 & NMinus1);
						bool flag(false);
						// 9.
						if (getGrid(origin_tp, t1) && dir != 3)
						{
							flag = true;
							dir = 9;
						}
						// 10.
						else if (getGrid(origin_tp, t2) && dir != 5)
						{
							flag = true;
							t1 = t2;
							dir = 10;
						}
						if (flag)
						{
							x1 = x2;
							setGridTo1(originT_tp, t1);
							origin = origin_tp;
							originT = gridOrigin(gridTraverse, x1 & NMinus1);
							kinkOrigin = gridKinksOrigin(x1 & NMinus1, 0);
							continue;
						}
					}
					scanf("%d", &x1);
				} while (((x - x1) & NMinus1) != 0 || t != t1);
				//until return to the beginning
				wdnm += abs((x - x1) / int(N));
				//printf("wdnm:%3u\n", wdnm);
			}
			//find the next beginning point
			else
			{
				//printf("cnt:%u\n", cnt);
				int s(startingBrick);
				unsigned int d;
				while (s < BrickNumTotal)
					if ((d = grid[s] ^ gridTraverse[s]) == 0)s++;
					else break;
				if (s == BrickNumTotal)return wdnm;
				startingBrick = s;
				x = s / BrickNum;
				t = ((s & BrickMinus1) << 5) + ffs(d);
			}
		}
#undef getState
	}
	//correlation
	void calCorrelation()
	{
		for (unsigned int c0(0); c0 < N; ++c0)
		{
			for (unsigned int c2(0); c2 < BrickNum; ++c2)
				tp0[c2] = grid[c0 * BrickNum + c2];
			for (unsigned int c1(0); c1 < N / 2; ++c1)
			{
				unsigned int s(0);
				unsigned int* origin(grid + ((c0 + c1 + 1) & NMinus1) * BrickNum);
				for (unsigned int c2(0); c2 < BrickNum; ++c2)
				{
					s += popc(tp0[c2] ^ origin[c2]);
				}
				correlation[c1] += s;
			}
		}
	}

	//one sample of h
	void getOneSample(float _h, unsigned int balanceLoops, unsigned int loops)
	{
		init(_h);
		reachBalance(balanceLoops);
#ifdef PRINTDEBUGINFO
		timer.begin();
#endif
		for (unsigned int c0(0); c0 < loops; ++c0)
			oneLoop();
#ifdef PRINTDEBUGINFO
		timer.end();
		timer.print("2000 loops:");
		printResults();
#endif
		//double cor(correlation[0]);
		//double avg(averageSpin);
		//double R(Rm);
		//cor /= (unsigned long long)loops * SpinNum;
		//avg /= (unsigned long long)loops * SpinNum;
		//R /= rounds;
		//cor = 1 - 2 * cor;
		//avg = 2 * avg - 1;
		//cor -= avg * avg;
		//printf("%f\t%f\n", abs(h), R);
		//printf("%f\t%f\t%f\t%f\n", abs(h), R, cor, avg);
	}

	//print one world line
	void print(unsigned int X)const
	{
		printBit(grid + X * BrickNum, N, 1);
	}
	//print the whole grid (1 + 1 dimension)
	void print()const
	{
		for (unsigned int c0(0); c0 < N; ++c0)
			printBit(grid + c0 * BrickNum, N, 1);
	}
	//print debug info
	void printDebug()
	{
		printf("Xi:%4d, Xm:%4d, Tm:%4u, Wind:%2u\n", Xi, Xm, Tm, singleWindingNumber());
	}
	//print statistical results
	void printResults()
	{
		printf("Rounds:%4u, Rm:%4u, Steps Per Loop:%.3f, R:%.6f\n",
			rounds, Rm, float(steps) / rounds, float(Rm) / rounds);
	}
	//print sim info
	void printSimInfo()const
	{
		printf("N:%4u, h:%4.3f, tau:%4u\n", N, h, tauA);
	}
#undef get
#undef set
#undef getGrid
#undef setGrid
#undef setGridTo1
};


template<unsigned int N, unsigned int SpaceDim = 1>
struct GridParameter
{
	Grid<N, SpaceDim>* grid;
	float h;
	unsigned int balanceLoops;
	unsigned int loops;
	unsigned int threadId;
	void run()
	{
		grid->getOneSample(h, balanceLoops, loops);
	}
};


int main()
{
	constexpr unsigned int N = 64;
	constexpr unsigned int SpaceDim = 1;
	using Grid = Grid<N, SpaceDim>;
	using GridParameter = GridParameter<N, SpaceDim>;

	unsigned int balanceLoops(10000);
	unsigned int loops(20000);
	float h0(0.01f), h1(2.f);
	//float h0(1.f - 0.001f * 15), h1(1.f + 0.001f * 16);
	constexpr unsigned int samples(128);
	float dh((h1 - h0) / (samples - 1));

	//ST
	/*Grid grid;
	grid.getOneSample(1, balanceLoops, loops);
	double cor;
	double avg(grid.averageSpin);
	avg /= (unsigned long long)loops * Grid::SpinNum;
	avg = 2 * avg - 1;
	for (unsigned int c0(0); c0 < N / 2; ++c0)
	{
		cor = grid.correlation[c0];
		cor /= (unsigned long long)loops * Grid::SpinNum;
		cor = 1 - 2 * cor - avg * avg;
		printf("%u\t%f\n", c0 + 1, cor);
	}
	grid.printResults();*/

	unsigned long long cor[samples][32];
	unsigned long long avg[samples];

	//MT
	unsigned long long threadNum;
#ifdef _WIN32
	SYSTEM_INFO systemInfo;
	GetSystemInfo(&systemInfo);
	threadNum = systemInfo.dwNumberOfProcessors;
#else
	threadNum = get_nprocs_conf();
#endif
	printf("ThreadNum:%3u\n", threadNum);

	Grid* grids((Grid*)malloc(threadNum * sizeof(Grid)));
	GridParameter* parameters((GridParameter*)
		malloc(threadNum * sizeof(GridParameter)));

#ifdef _WIN32
	HANDLE* threads(nullptr);
#else
	pthread_t* threads(nullptr);
#endif
#ifdef _WIN32
	threads = (HANDLE*)::malloc(threadNum * sizeof(HANDLE));
#else
	threads = (pthread_t*)::malloc((threadNum - 1) * sizeof(pthread_t));
#endif

	//ptr:	GridParameter
#ifdef _WIN32
	void (*lambda)(void*) = [](void* ptr)
#else
	void* (*lambda)(void*) = [](void* ptr)->void*
#endif
	{
		((GridParameter*)ptr)->run();
#ifdef _WIN32
		//_endthread();
#else
		return 0;
#endif
	};

	for (unsigned int c0(0); c0 < threadNum; ++c0)
	{
		new(grids + c0)Grid(c0);
		parameters[c0].grid = grids + c0;
		parameters[c0].loops = loops;
		parameters[c0].balanceLoops = balanceLoops;
		parameters[c0].threadId = c0;
	}
	for (unsigned int c0(0); c0 < samples; c0 += threadNum)
	{
		unsigned int c1(0);
		for (; c1 < threadNum; ++c1)
		{
			parameters[c1].h = h0 + (c0 + c1) * dh;
#ifdef _WIN32
			threads[c1] = (HANDLE)_beginthread(lambda, 0, parameters + c1);
#else
			pthread_create(threads + c1, 0, lambda, paras + c1);
			pthread_detach(threads[c1]);
#endif
		}
#ifdef _WIN32
		DWORD rc = WaitForMultipleObjects(threadNum, threads, true, INFINITE);
		for (unsigned int c1(0); c1 < threadNum; ++c1)
		{
			avg[c0 + c1] = grids[c1].averageSpin;
			for (unsigned int c2(0); c2 < N / 2; ++c2)
				cor[c0 + c1][c2] = grids[c1].correlation[c2];
		}
		//if (rc == WAIT_OBJECT_0)
		//{
			//printf("All thread terminite\n");
			//for (unsigned long long c0(0); c0 < threadNum - 1; ++c0)
			//	CloseHandle(threads[c0]);
		//}
#else
		for (unsigned long long c0(0); c0 < threadNum - 1; ++c0)
			pthread_join(threads[c0], nullptr);
#endif
	}

	for (unsigned int c0(0); c0 < threadNum; ++c0)
		grids[c0].~Grid();

	for (unsigned int c0(0); c0 < samples; ++c0)
	{
		double _cor;
		double _avg(avg[c0]);
		_avg /= (unsigned long long)loops * Grid::SpinNum;
		_avg = 2 * _avg - 1;
		printf("%f\t%f\t", h0 + c0 * dh, _avg);
		for (unsigned int c1(0); c1 < N / 2; ++c1)
		{
			_cor = cor[c0][c1];
			_cor /= (unsigned long long)loops * Grid::SpinNum;
			_cor = 1 - 2 * _cor - _avg * _avg;
			printf("%f\t", _cor);
		}
		printf("\n");
	}

	free(grids);
	free(parameters);
	free(threads);
}
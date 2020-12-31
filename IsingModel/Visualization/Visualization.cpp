#include <cstdio>
#include <random>
#include <_Bit.h>
#include <_Math.h>
#include <_Time.h>
#include <GL/_Window.h>
#ifdef _WIN32
#include <intrin.h>
#define popc(x) __popcnt(x)
#define ffs(x) _tzcnt_u32(x)
#else
// #include <immintrin.h>
#define popc(x) __builtin_popcount(x)
#endif
//#define PRINTDEBUGINFO

//for space-dim = 1
constexpr unsigned int powd(unsigned int a, unsigned int n)
{
	if (n)return powd(a, n - 1) * a;
	else return 1;
}

//constexpr unsigned int N(64);
//constexpr unsigned int SpaceDim(1);
//constexpr unsigned int BrickNum(N / 32);
//constexpr unsigned int BrickMinus1(BrickNum - 1);
//constexpr unsigned int BrickNumTotal(BrickNum* powd(N, SpaceDim));
//constexpr unsigned int SpinNum(powd(N, SpaceDim + 1));
//constexpr unsigned int GridSize(powd(N, SpaceDim)* BrickNum * sizeof(unsigned int));
//constexpr unsigned int GridUSize(powd(N, SpaceDim) * sizeof(unsigned int));
//constexpr unsigned int GridKinksSize(powd(N, SpaceDim)* SpaceDim* BrickNum * sizeof(unsigned int));
//constexpr unsigned int GridKinksNumSize(powd(N, SpaceDim)* SpaceDim * sizeof(unsigned int));



//Each Grid for each thread
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

	static constexpr float h = 1.5f;

#define IF(x) (x)?
#define THEN(x) (x):
#define ELSE(x) (x)
	static constexpr unsigned int tauA = 4;
		//IF(1.f / h < 4)
		//THEN(4)
		//ELSE(
		//	IF(1.f / h > N / 2)
		//	THEN(N / 2)
		//	ELSE((unsigned int)(1.f / h) & (-2)));		//\tau_a
#undef IF
#undef THEN
#undef ELSE
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
	static constexpr float PaTauAInv = 1 / PaTauA;

	const std::uniform_real_distribution<float> rd;
	const std::uniform_int_distribution<unsigned int> rdint;
	const std::uniform_int_distribution<unsigned int> rdtauA;
	const std::uniform_int_distribution<unsigned int> rdWorldLineDim;
	const std::uniform_int_distribution<int> rdWorldLineDir;
	//if return non-negetive, then plus 1 to make sure the result is in [-tauBD2, tauBD2]\{0}
	const std::uniform_int_distribution<int> rdDeltaB;
	//if return non-negetive, then plus 1 to make sure the result is in [-tauCD2, tauCD2]\{0}
	const std::uniform_int_distribution<int> rdDeltaC;
	const std::uniform_int_distribution<int> rdDeltaC1;

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

	Grid()
		:
		rd(0, 1),
		rdint(0, N - 1),
		rdtauA(1, tauA),
		rdWorldLineDim(0, SpaceDim - 1),
		rdWorldLineDir(0, 1),
		rdDeltaB(-int(tauBD2), tauBD2 - 1),
		rdDeltaC(-int(tauCD2), tauCD2 - 1),
		rdDeltaC1(-int(tauCD2), tauCD2),
		mt(time(nullptr)),
		grid((unsigned int*)malloc(GridSize)),
		gridU((unsigned int*)malloc(GridUSize)),
		gridKinks((unsigned int*)malloc(GridKinksSize)),
		gridTraverse((unsigned int*)malloc(GridSize)),
		operation(OperationType::None)
	{
		clear();
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
		Ti = -1;
		Rm = 0;
		rounds = 0;
		steps = 0;
		deltaMove = 0;
		deltaInsert = 0;
		deltaDelete = 0;
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
				Rm += (traverse() != 0);
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
	}
	//reach balance
	void reachBalance()
	{
		printSimInfo();
		for (unsigned int c0(0); c0 < 5000; ++c0)
		{
			do
			{
				operate();
			} while (operation != OperationType::Annihilate);
			printResults();
		}
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
		memset(gridTraverse, 0, GridSize);
#undef getState
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
		printf("N:%4u h:%.3f\n", N, h);
	}
#undef get
#undef set
#undef getGrid
#undef setGrid
#undef setGridTo1
};

namespace OpenGL
{
	template<unsigned int N, unsigned int SpaceDim = 1>
	struct VisualGrid :OpenGL
	{
		// Points of Spins
		template<unsigned int N, unsigned int SpaceDim = 1>
		struct Spins
		{
			static constexpr size_t SpinsSize = powd(N, SpaceDim + 1) * sizeof(Math::vec2<float>);
			Math::vec2<float>* positions;
			Spins(Math::vec2<float> origin, Math::vec2<float> ending)
				:
				positions((Math::vec2<float>*)malloc(SpinsSize))
			{
				float deltaT{ (ending.data[0] - origin.data[0]) / N };
				float deltaX{ (ending.data[1] - origin.data[1]) / N };
				for (unsigned int c0(0); c0 < N; ++c0)
					for (unsigned int c1(0); c1 < N; ++c1)
						positions[c0 * N + c1] = origin + Math::vec2<float>{c1* deltaT, c0* deltaX};
			}
			~Spins()
			{
				free(positions);
				positions = nullptr;
			}
		};
		// Lines along with T
		template<unsigned int N, unsigned int SpaceDim = 1>
		struct SpinLines
		{
			static constexpr size_t LinesSize = powd(N, SpaceDim + 1) * 2 * sizeof(Math::vec2<float>);
			Math::vec2<float>* positions;
			SpinLines(Math::vec2<float> origin, Math::vec2<float> ending)
				:
				positions((Math::vec2<float>*)malloc(LinesSize))
			{
				float deltaT{ (ending.data[0] - origin.data[0]) / N };
				float deltaX{ (ending.data[1] - origin.data[1]) / N };
				for (unsigned int c0(0); c0 < N; ++c0)
					for (unsigned int c1(0); c1 < N; ++c1)
					{
						positions[2 * (c0 * N + c1) + 0] = origin + Math::vec2<float>{(c1 + 0)* deltaT, c0* deltaX};
						positions[2 * (c0 * N + c1) + 1] = origin + Math::vec2<float>{(c1 + 1)* deltaT, c0* deltaX};
					}
			}
			~SpinLines()
			{
				free(positions);
				positions = nullptr;
			}
		};
		// Lines along with X
		template<unsigned int N, unsigned int SpaceDim = 1>
		struct KinkLines
		{
			static constexpr size_t LinesSize = powd(N, SpaceDim + 1) * 2 * sizeof(Math::vec2<float>);
			Math::vec2<float>* positions;
			KinkLines(Math::vec2<float> origin, Math::vec2<float> ending)
				:
				positions((Math::vec2<float>*)malloc(LinesSize))
			{
				float deltaT{ (ending.data[0] - origin.data[0]) / N };
				float deltaX{ (ending.data[1] - origin.data[1]) / N };
				origin.data[0] += 0.5f * deltaT;
				for (unsigned int c0(0); c0 < N; ++c0)
					for (unsigned int c1(0); c1 < N; ++c1)
					{
						positions[2 * (c0 * N + c1) + 0] = origin + Math::vec2<float>{c1* deltaT, (c0 + 0)* deltaX};
						positions[2 * (c0 * N + c1) + 1] = origin + Math::vec2<float>{c1* deltaT, (c0 + 1)* deltaX};
					}
			}
			~KinkLines()
			{
				free(positions);
				positions = nullptr;
			}
		};
		template<unsigned int N, unsigned int SpaceDim = 1>
		struct SpinsData :Buffer::Data
		{
			static constexpr size_t SpinsSize = powd(N, SpaceDim + 1) * sizeof(Math::vec2<float>);
			Spins<N, SpaceDim>* spins;
			SpinsData(Spins<N, SpaceDim>* _spins)
				:
				Data(StaticDraw),
				spins(_spins)
			{
			}
			virtual void* pointer()override
			{
				return spins->positions;
			}
			virtual unsigned int size()override
			{
				return SpinsSize;
			}
		};
		template<unsigned int N, unsigned int SpaceDim = 1>
		struct SpinLinesData :Buffer::Data
		{
			static constexpr size_t LinesSize = powd(N, SpaceDim + 1) * 2 * sizeof(Math::vec2<float>);
			SpinLines<N, SpaceDim>* spinLines;
			SpinLinesData(SpinLines<N, SpaceDim>* _spinLines)
				:
				Data(StaticDraw),
				spinLines(_spinLines)
			{
			}
			virtual void* pointer()override
			{
				return spinLines->positions;
			}
			virtual unsigned int size()override
			{
				return LinesSize;
			}
		};
		template<unsigned int N, unsigned int SpaceDim = 1>
		struct KinkLinesData :Buffer::Data
		{
			static constexpr size_t LinesSize = powd(N, SpaceDim + 1) * 2 * sizeof(Math::vec2<float>);
			KinkLines<N, SpaceDim>* kinkLines;
			KinkLinesData(KinkLines<N, SpaceDim>* _kinkLines)
				:
				Data(StaticDraw),
				kinkLines(_kinkLines)
			{
			}
			virtual void* pointer()override
			{
				return kinkLines->positions;
			}
			virtual unsigned int size()override
			{
				return LinesSize;
			}
		};
		template<unsigned int N, unsigned int SpaceDim = 1>
		struct EndPointsData :Buffer::Data
		{
			static constexpr unsigned int NMinus1 = N - 1;
			Grid<N, SpaceDim>* grid;
			Math::vec2<float>data[2];
			Math::vec2<float> origin;
			float deltaT;
			float deltaX;
			EndPointsData(Grid<N, SpaceDim>* _grid, Math::vec2<float> _origin, Math::vec2<float> ending)
				:
				grid(_grid),
				origin(_origin),
				deltaT((ending.data[0] - _origin.data[0]) / N),
				deltaX((ending.data[1] - _origin.data[1]) / N)
			{
				origin.data[0] += 0.5f * deltaT;
			}
			virtual void* pointer()override
			{
				data[0] = origin + Math::vec2<float>{grid->Ti* deltaT, (grid->Xi& NMinus1)* deltaX};
				data[1] = origin + Math::vec2<float>{grid->Tm* deltaT, (grid->Xm& NMinus1)* deltaX};
				return &data;
			}
			virtual unsigned int size()override
			{
				return sizeof(data);
			}
		};


		template<unsigned int N, unsigned int SpaceDim = 1>
		struct GridSpinsData :Buffer::Data
		{
			static constexpr unsigned int BrickNum = N / 32;
			static constexpr unsigned int GridSize = powd(N, SpaceDim) * BrickNum * sizeof(unsigned int);
			unsigned int* grid;

			GridSpinsData(Grid<N, SpaceDim>* _grid)
				:
				grid(_grid->grid)
			{
			}
			virtual void* pointer()override
			{
				return grid;
			}
			virtual unsigned int size()override
			{
				return GridSize;
			}
		};
		template<unsigned int N, unsigned int SpaceDim = 1>
		struct GridKinksData :Buffer::Data
		{
			static constexpr unsigned int BrickNum = N / 32;
			static constexpr unsigned int GridKinksSize = powd(N, SpaceDim) * SpaceDim * BrickNum * sizeof(unsigned int);
			unsigned int* gridKinks;

			GridKinksData(Grid<N, SpaceDim>* _grid)
				:
				gridKinks(_grid->gridKinks)
			{
			}
			virtual void* pointer()override
			{
				return gridKinks;
			}
			virtual unsigned int size()override
			{
				return GridKinksSize;
			}
		};


		template<unsigned int N, unsigned int SpaceDim = 1>
		struct SpinsRenderer :Program
		{
			BufferConfig spinsArray;
			VertexAttrib positions;

			SpinsRenderer(SourceManager* _sm, Buffer* _spinsBuffer)
				:
				Program(_sm, "SpinsRenderer", Vector<VertexAttrib*>{&positions}),
				spinsArray(_spinsBuffer, ArrayBuffer),
				positions(&spinsArray, 0, VertexAttrib::two, VertexAttrib::Float, false, sizeof(Math::vec2<float>), 0, 0)
			{
				init();
			}
			virtual void initBufferData()override
			{
			}
			virtual void run()override
			{
				glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
				glClear(GL_COLOR_BUFFER_BIT);
				glPointSize(5);
				glDrawArrays(GL_POINTS, 0, N * N);
			}
		};
		template<unsigned int N, unsigned int SpaceDim = 1>
		struct SpinLinesRenderer :Program
		{
			BufferConfig spinLinesArray;
			VertexAttrib positions;

			SpinLinesRenderer(SourceManager* _sm, Buffer* _spinLinesBuffer)
				:
				Program(_sm, "SpinLinesRenderer", Vector<VertexAttrib*>{&positions}),
				spinLinesArray(_spinLinesBuffer, ArrayBuffer),
				positions(&spinLinesArray, 0, VertexAttrib::two, VertexAttrib::Float, false, sizeof(Math::vec2<float>), 0, 0)
			{
				init();
			}
			virtual void initBufferData()override
			{
			}
			virtual void run()override
			{
				glDrawArrays(GL_LINES, 0, N * N * 2);
			}
		};
		template<unsigned int N, unsigned int SpaceDim = 1>
		struct KinkLinesRenderer :Program
		{
			BufferConfig kinkLinesArray;
			VertexAttrib positions;

			KinkLinesRenderer(SourceManager* _sm, Buffer* _kinkLinesBuffer)
				:
				Program(_sm, "KinkLinesRenderer", Vector<VertexAttrib*>{&positions}),
				kinkLinesArray(_kinkLinesBuffer, ArrayBuffer),
				positions(&kinkLinesArray, 0, VertexAttrib::two, VertexAttrib::Float, false, sizeof(Math::vec2<float>), 0, 0)
			{
				init();
			}
			virtual void initBufferData()override
			{
			}
			virtual void run()override
			{
				glDrawArrays(GL_LINES, 0, N * N * 2);
			}
		};
		template<unsigned int N, unsigned int SpaceDim = 1>
		struct EndPointRenderer :Program
		{
			BufferConfig endPointArray;
			VertexAttrib positions;

			EndPointRenderer(SourceManager* _sm, Buffer* _endPointBuffer)
				:
				Program(_sm, "EndPointsRenderer", Vector<VertexAttrib*>{&positions}),
				endPointArray(_endPointBuffer, ArrayBuffer),
				positions(&endPointArray, 0, VertexAttrib::two, VertexAttrib::Float, false, sizeof(Math::vec2<float>), 0, 0)
			{
				init();
			}
			virtual void initBufferData()override
			{
			}
			virtual void run()override
			{
				endPointArray.refreshData();
				glPointSize(11);
				glDrawArrays(GL_POINTS, 0, 2);
			}
		};


		SourceManager sm;
		Grid<N, SpaceDim> grid;

		Math::vec2<float> origin;
		Math::vec2<float> ending;

		Spins<N, 1> spins;
		SpinLines<N, 1> spinLines;
		KinkLines<N, 1> kinkLines;

		SpinsData<N, SpaceDim> spinsData;
		SpinLinesData<N, SpaceDim> spinLinesData;
		KinkLinesData<N, SpaceDim> kinkLinesData;
		EndPointsData<N, SpaceDim> endPointsData;

		GridSpinsData<N, SpaceDim> gridSpinsData;
		GridKinksData<N, SpaceDim> gridKinksData;


		Buffer spinsBuffer;
		Buffer spinLinesBuffer;
		Buffer kinkLinesBuffer;
		Buffer endPointsBuffer;

		Buffer gridSpinsBuffer;
		Buffer gridKinksBuffer;

		BufferConfig gridSpinsStorage;
		BufferConfig gridKinksStorage;

		SpinsRenderer<N, SpaceDim> spinsRenderer;
		SpinLinesRenderer<N, SpaceDim> spinLinesRenderer;
		KinkLinesRenderer<N, SpaceDim> kinkLinesRenderer;
		EndPointRenderer<N, SpaceDim> endPointsRenderer;

		int update;

		VisualGrid()
			:
			sm(),
			grid(),

			origin{ -0.99f, -0.99f },
			ending{ 0.99f, 0.99f },

			spins(origin, ending),
			spinLines(origin, ending),
			kinkLines(origin, ending),

			spinsData(&spins),
			spinLinesData(&spinLines),
			kinkLinesData(&kinkLines),
			endPointsData(&grid, origin, ending),

			gridSpinsData(&grid),
			gridKinksData(&grid),

			spinsBuffer(&spinsData),
			spinLinesBuffer(&spinLinesData),
			kinkLinesBuffer(&kinkLinesData),
			endPointsBuffer(&endPointsData),

			gridSpinsBuffer(&gridSpinsData),
			gridKinksBuffer(&gridKinksData),

			gridSpinsStorage(&gridSpinsBuffer, ShaderStorageBuffer, 1),
			gridKinksStorage(&gridKinksBuffer, ShaderStorageBuffer, 2),

			spinsRenderer(&sm, &spinsBuffer),
			spinLinesRenderer(&sm, &spinLinesBuffer),
			kinkLinesRenderer(&sm, &kinkLinesBuffer),
			endPointsRenderer(&sm, &endPointsBuffer),

			update(-1)
		{
		}
		void updateGrid()
		{
			if (update == -1)
			{
				grid.reachBalance();
				update = 0;
			}
			/*if (grid.operation == Grid::OperationType::Annihilate)
			{
				grid.timer.begin();
				unsigned int warpingNum(grid.traverse());
				grid.timer.end();
				grid.timer.print();
				printf("Winding Number: %u\n", warpingNum);
				update = false;
				grid.operation = Grid::OperationType::None;
			}*/
			if (update)
			{
				grid.timer.begin();
				for (unsigned int c0(0); c0 < 1; ++c0)
				{
					do
					{
						grid.operate();
					} while (grid.operation != Grid<N, SpaceDim>::OperationType::Annihilate);

					//if ((n++ & 1023) == 0)
						//grid.printResults();
				}
				grid.timer.end();
				grid.timer.print("1000 Loops:");

				grid.timer.begin();
				grid.traverse();
				grid.timer.end();
				grid.timer.print("Traverse:");
				//update = false;

#ifdef PRINTDEBUGINFO
				/*printf("DeltaMove:\t%.3f\nDeltaInsert:\t%.3f\nDeltaDelete:\t%.3f\n",
					float(grid.deltaMove) / grid.steps,
					float(grid.deltaInsert) / grid.steps,
					float(grid.deltaDelete) / grid.steps);*/
#endif
					//grid.simpleTestSet();

				//update = false;
			}
			gridSpinsStorage.refreshData();
			gridKinksStorage.refreshData();
		}
		virtual void init(FrameScale const& _size)override
		{
			glViewport(0, 0, _size.w, _size.h);
			glDisable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glLineWidth(3);
			//trans.init(_size);
			//renderer.transUniform.dataInit();
			//renderer.particlesArray.dataInit();
			//computeParticles.init();
			spinsRenderer.spinsArray.dataInit();
			spinLinesRenderer.spinLinesArray.dataInit();
			kinkLinesRenderer.kinkLinesArray.dataInit();
			endPointsRenderer.endPointArray.dataInit();
			gridSpinsStorage.dataInit();
			gridKinksStorage.dataInit();
		}
		virtual void run()override
		{
			spinsRenderer.use();
			spinsRenderer.run();
			spinLinesRenderer.use();
			spinLinesRenderer.run();
			kinkLinesRenderer.use();
			kinkLinesRenderer.run();
			if (grid.Ti >= 0)
			{
				endPointsRenderer.use();
				endPointsRenderer.run();
			}
		}
		virtual void frameSize(int _w, int _h) override
		{
			//trans.resize(_w, _h);
			glViewport(0, 0, _w, _h);
		}
		virtual void framePos(int, int) override
		{
		}
		virtual void frameFocus(int) override
		{
		}
		virtual void mouseButton(int _button, int _action, int _mods) override
		{
			switch (_button)
			{
				//case GLFW_MOUSE_BUTTON_LEFT:trans.mouse.refreshButton(0, _action); break;
				//case GLFW_MOUSE_BUTTON_MIDDLE:trans.mouse.refreshButton(1, _action); break;
				//case GLFW_MOUSE_BUTTON_RIGHT:trans.mouse.refreshButton(2, _action); break;
			}
		}
		virtual void mousePos(double _x, double _y) override
		{
			//trans.mouse.refreshPos(_x, _y);
		}
		virtual void mouseScroll(double _x, double _y)override
		{
			//if (_y != 0.0)
				//trans.scroll.refresh(_y);
		}
		virtual void key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods) override
		{
			switch (_key)
			{
			case GLFW_KEY_ESCAPE:
				if (_action == GLFW_PRESS)
					glfwSetWindowShouldClose(_window, true);
				break;
			case GLFW_KEY_SPACE:
				if (_action == GLFW_PRESS || _action == GLFW_REPEAT)
					update = 1;
				break;
				//case GLFW_KEY_D:trans.key.refresh(1, _action); break;
				//case GLFW_KEY_W:trans.key.refresh(2, _action); break;
				//case GLFW_KEY_S:trans.key.refresh(3, _action); break;
			}
		}
	};
}

int main()
{
	OpenGL::OpenGLInit init(4, 5);
	Window::Window::Data winParameters
	{
		"Worm Algorithm",
		{
			{1800,1800},
			true,false
		}
	};
	Window::WindowManager wm(winParameters);
	OpenGL::VisualGrid<64, 1> visualGrid;
	wm.init(0, &visualGrid);
	init.printRenderer();
	glfwSwapInterval(0);
	FPS fps;
	fps.refresh();
	while (!wm.close())
	{
		wm.pullEvents();
		wm.render();
		wm.swapBuffers();
		visualGrid.updateGrid();
		fps.refresh();
		//::printf("\r%.2lf\t", fps.fps);
		//fps.printFPS(1);
	}
	return 0;
}
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <_Time.h>
#include <_Bit.h>
#include <random>
#ifdef _WIN32
#include <intrin.h>
#define popc(x) __popcnt(x)
#else
// #include <immintrin.h>
#define popc(x) __builtin_popcount(x)
#endif

//dimension: 1 + 1
//size: N = 128
//use int32 instead of int64 because CUDA's support for int64 is simulated


constexpr unsigned int powd(unsigned int a, unsigned int n)
{
	if (n)return powd(a, n - 1) * a;
	else return 1;
}

constexpr unsigned int N(64);
constexpr unsigned int NMinus1(N - 1);
constexpr unsigned int BrickNum(N / 32);
constexpr unsigned int BrickMinus1(BrickNum - 1);
constexpr unsigned int SpaceDim(1);
constexpr unsigned int GridSize(powd(N, SpaceDim)* BrickNum * sizeof(unsigned int));
constexpr unsigned int GridUSize(powd(N, SpaceDim) * sizeof(unsigned int));
constexpr unsigned int GridKinksSize(powd(N, SpaceDim)* SpaceDim* BrickNum * sizeof(unsigned int));
constexpr unsigned int GridKinksNumSize(powd(N, SpaceDim)* SpaceDim * sizeof(unsigned int));

constexpr float Pa(1.f / 3);
constexpr float Pb(1.f / 3);
constexpr float Pc(1.f / 6);
constexpr float APa(Pa);
constexpr float APb(Pa + Pb);
constexpr float APc0(Pa + Pb + Pc);

constexpr float h(0.1f);
constexpr unsigned int tauA(10);//\tau_a
constexpr unsigned int tauAD2(tauA / 2);//\dfrac{\tau_a}{2}
constexpr unsigned int tauB(10);//\tau_b
constexpr unsigned int tauBM1(tauB - 1);//\tau_b-1
constexpr unsigned int tauBD2(tauB / 2);//\dfrac{\tau_b}{2}
constexpr unsigned int tauBD2M1(tauBD2 - 1);//\dfrac{\tau_b}{2}-1
constexpr unsigned int tauC(10);//\tau_c
constexpr unsigned int tauCM1(tauC - 1);//\tau_c-1
constexpr unsigned int tauCD2(tauC / 2);//\dfrac{\tau_c}{2}
constexpr unsigned int tauCD2M1(tauCD2 - 1);//\dfrac{\tau_c}{2}-1
constexpr float tauCInv(1.f / tauC);
constexpr float PaTauA(Pa* tauA);
constexpr float PaTauAInv(1 / PaTauA);



std::uniform_real_distribution<float> rd(0, 1);
std::uniform_int_distribution<unsigned int> rdint(0, N - 1);
std::uniform_int_distribution<unsigned int> rdtauA(1, tauA);
std::uniform_int_distribution<unsigned int> rdWorldLineDim(0, SpaceDim - 1);
std::uniform_int_distribution<int> rdWorldLineDir(0, 1);
//if return non-negetive, then plus 1 to make sure the result is in [-tauBD2, tauBD2)\{0}
std::uniform_int_distribution<int> rdDeltaB(-int(tauBD2), tauBD2 - 2);
//if return non-negetive, then plus 1 to make sure the result is in [-tauCD2, tauCD2)\{0}
std::uniform_int_distribution<int> rdDeltaC(-int(tauCD2), tauCD2 - 2);
std::uniform_int_distribution<int> rdDeltaC1(-int(tauCD2), tauCD2 - 1);

//Each Grid for each thread
struct Grid
{
	std::mt19937 mt;

	enum OperationType
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
	unsigned int* gridKinksTraverse;

	int Ti, Tm;//-1 means not created, does not need modulo
	int Xi, Xm;

	OperationType operation;

	static unsigned int Rm;
	static unsigned int rounds;
	static unsigned int steps;

	unsigned int tp0[BrickNum];
	unsigned int tp1[BrickNum];

#define get(x, y) (((x) >> (y)) & 1)
#define set(x,y) ((x) ^= (1u << (y)))

	Grid()
		:
		mt(time(nullptr)),
		grid((unsigned int*)malloc(GridSize)),
		gridU((unsigned int*)malloc(GridUSize)),
		gridKinks((unsigned int*)malloc(GridKinksSize)),
		gridTraverse((unsigned int*)malloc(GridSize)),
		gridKinksTraverse((unsigned int*)malloc(GridKinksSize)),
		operation(None)
	{
		clear();
	}
	~Grid()
	{
		free(grid);
		free(gridU);
		free(gridKinks);
		free(gridTraverse);
		free(gridKinksTraverse);
	}
	//Clear the grid to original state
	void clear()
	{
		memset(grid, 0, GridSize);
		memset(gridU, 0, GridUSize);
		memset(gridKinks, 0, GridKinksSize);
		memset(gridTraverse, 0, GridSize);
		memset(gridKinksTraverse, 0, GridKinksSize);
		Ti = -1;
	}
	inline unsigned int* gridKinksOrigin(unsigned int X, unsigned int dim)
	{
		return gridKinks + BrickNum * (dim + SpaceDim * X);
	}
	//logic or on section [t0, t1] in different lins of gridKinks, stores in tp0
	void kinksLogicOr(int Xk, int Xk1, int Xk2, unsigned int t0, unsigned int t1)
	{
		unsigned int* origin(gridKinksOrigin(Xk, 0));
		unsigned int* origin1(gridKinksOrigin(Xk1, 0));
		unsigned int* origin2(gridKinksOrigin(Xk2, 0));
		unsigned int Ua(t0 >> 5);
		unsigned int Ub(t1 >> 5);
		if (t0 <= t1)
		{
			tp0[Ua] = origin[Ua] | origin1[Ua] | origin2[Ua];
			if (Ua != Ub)
			{
				tp0[Ua] = origin[Ua] | origin1[Ua] | origin2[Ua];
				for (unsigned int c0(Ua + 1); c0 < Ub; ++c0)
					tp0[c0] = origin[c0] | origin1[c0] | origin2[c0];
				tp0[Ub] = origin[Ub] | origin1[Ub] | origin2[Ub];
			}
		}
		else
		{
			tp0[Ua] = origin[Ua] | origin1[Ua] | origin2[Ua];
			if (Ua != Ub)
				tp0[Ub] = origin[Ub] | origin1[Ub] | origin2[Ub];
			for (unsigned int c0(Ua + 1); c0 < BrickNum + Ub; ++c0)
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
		unsigned int* gridOrigin(grid + BrickNum * X);
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
				U += popc(gridOrigin[Ua] & mask);
				tp[Ua] = gridOrigin[Ua] ^ mask;
			}
			else
			{
				unsigned int mask(0xffffffff);
				mask <<= Da;
				U += popc(gridOrigin[Ua] & mask);
				tp[Ua] = gridOrigin[Ua] ^ mask;
				for (unsigned int c0(Ua + 1); c0 < Ub; ++c0)
				{
					U += popc(gridOrigin[c0]);
					tp[c0] = gridOrigin[c0] ^ 0xffffffff;
				}
				mask = 0xffffffff;
				mask >>= (31 - Db);
				U += popc(gridOrigin[Ub] & mask);
				tp[Ub] = gridOrigin[Ub] ^ mask;
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
				U += popc(gridOrigin[Ua] & mask);
				tp[Ua] = gridOrigin[Ua] ^ mask;
			}
			else
			{
				unsigned int mask(0xffffffff);
				mask <<= Da;
				U += popc(gridOrigin[Ua] & mask);
				tp[Ua] = gridOrigin[Ua] ^ mask;

				mask = 0xffffffff;
				mask >>= (31 - Db);
				U += popc(gridOrigin[Ub] & mask);
				tp[Ub] = gridOrigin[Ub] ^ mask;
			}
			for (unsigned int c0(Ua + 1); c0 < BrickNum + Ub; ++c0)
			{
				unsigned int p(c0 & BrickMinus1);
				U += popc(gridOrigin[p]);
				tp[p] = gridOrigin[p] ^ 0xffffffff;
			}
		}
		return U;
	}
	//copy the states changed by flipSpins
	//parameters should be the same as flipSpins
	void copySpins(unsigned int* tp, unsigned int X, unsigned int t0, unsigned int t1)
	{
		unsigned int* gridOrigin(grid + BrickNum * X);
		unsigned int Ua(t0 >> 5);
		unsigned int Ub(t1 >> 5);
		if (t0 <= t1)
		{
			gridOrigin[Ua] = tp[Ua];
			if (Ua != Ub)
			{
				for (unsigned int c0(Ua + 1); c0 < Ub; ++c0)
					gridOrigin[c0] = tp[c0];
				gridOrigin[Ub] = tp[Ub];
			}
		}
		else
		{
			gridOrigin[Ua] = tp[Ua];
			if (Ua != Ub)gridOrigin[Ub] = tp[Ub];
			for (unsigned int c0(Ua + 1); c0 < BrickNum + Ub; ++c0)
			{
				unsigned int p(c0 & BrickMinus1);
				gridOrigin[p] = tp[p];
			}
		}
	}
	//calculate the sum of all spins along with X
	unsigned int spinUpNum(unsigned int X)
	{
		unsigned int a(0);
		unsigned int* gridOrigin(grid + BrickNum * X);
		for (unsigned int c0(0); c0 < BrickNum; ++c0)
			a += popc(gridOrigin[c0]);
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
		printf("Create\t\t");
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
			operation = Create;
			printf("accepted\t");
			printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u\n", Xi, Xm, Ti, Tm);
		}
		else
		{
			Ti = -1;
			printf("denied\n");
		}
	}
	//annihilate defectes
	void annihilateDefects()
	{
		printf("Annihilate\t");
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
				Rm += (windingNumber() != 0);
				operation = Annihilate;
				printf("accepted\t");
				printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u\n", Xi, Xm, Ti, Tm);
				flag = true;
			}
		}
		if (!flag)printf("denied\n");
	}
	//move Tm
	void moveMT()
	{
		printf("Move\t\t");
		bool flag(false);
		unsigned int Tn;
		unsigned int t0, t1;
		int dd(rdDeltaB(mt));
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
		int dU(flipSpins(tp0, Xm & NMinus1, (t0 + 1) & NMinus1, t1));
		dd -= 2 * dU;//variation of U
		float acceptance(expf(h * dd));
		if (rd(mt) < acceptance)
		{
			copySpins(tp0, Xm & NMinus1, (t0 + 1) & NMinus1, t1);
			gridU[Xm & NMinus1] += dd;
			Tm = Tn;
			operation = Move;
			printf("accepted\t");
			printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u\n", Xi, Xm, Ti, Tm);
			flag = true;
		}
		if (!flag)printf("denied\n");
	}
	//insert a kink
	void insertKink()
	{
		printf("Insert\t\t");
		bool flag(false);
		int dim(rdWorldLineDim(mt));//which dimension that the movement takes 
		int dir(rdWorldLineDir(mt));//direction of the movement, 0 means -1, 1 means 1
		int Xn(Xm + (2 * dir - 1));//neighbour of Xm, new Xm
		int Xk((dir ? Xm : Xn) & NMinus1);//choose the index of gridKinks
		unsigned int Tn;//position of the add kink
		unsigned int* origin(gridKinksOrigin(Xk, dim));
		unsigned int t0, t1;
		int dd0, dd1;
		//kinks number in [Tm - tauCD2, Tm + tauCD2 - 1)
		unsigned int kn(kinksNum(origin, (Tm - tauCD2) & NMinus1, (Tm + tauCD2M1) & NMinus1));
		//make sure that there is space for a new kink
		if (tauCM1 != kn - get(origin[Tm >> 5], Tm & 31))
			do
			{
				//choose a position which has no kink
				dd0 = rdDeltaC(mt);
				if (dd0 >= 0)
				{
					dd0++;
					Tn = (Tm + dd0) & NMinus1;
					t0 = (Tm + 1) & NMinus1, t1 = Tn;
				}
				else
				{
					Tn = (Tm + dd0) & NMinus1;
					t0 = (Tn + 1) & NMinus1, t1 = Tm;
				}
			} while (get(origin[Tn >> 5], Tn & 31));
		else
		{
			printf("denied\n");
			return;
		}
		dd1 = dd0;
		int dU0(flipSpins(tp0, Xm & NMinus1, t0, t1));
		dd0 -= 2 * dU0;//variation of U
		int dU1(flipSpins(tp1, Xn & NMinus1, t0, t1));
		dd1 -= 2 * dU1;//variation of U
		float acceptance((tauC * expf(h * (dd0 + dd1)) / (kn + 1)));
		if (rd(mt) < acceptance)
		{
			copySpins(tp0, Xm & NMinus1, t0, t1);
			copySpins(tp1, Xn & NMinus1, t0, t1);
			set(origin[Tn >> 5], Tn & 31);
			gridU[Xm & NMinus1] += dd0;
			gridU[Xn & NMinus1] += dd1;
			Xm = Xn;
			operation = Insert;
			printf("accepted\t");
			printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u, Tn: %4u\n", Xi, Xm, Ti, Tm, Tn);
			flag = true;
		}
		if (!flag)printf("denied\n");
	}
	//delete a kink
	void deleteKink()
	{
		printf("Delete\t\t");
		bool flag(false);
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
		//make sure that n_k > 0
		if (kn)
			do
			{
				//choose a position which has a kink
				dd = rdDeltaC1(mt);
				Tn = (Tm + dd) & NMinus1;
				if (dd >= 0)t0 = Tm, t1 = Tn;
				else t0 = Tn, t1 = Tm;
			} while (!get(origin[Tn >> 5], Tn & 31));
		else
		{
			printf("denied\n");
			return;
		}
		float acceptance(kn * tauCInv);
		if (dd)
		{
			dd1 = dd0 = dd;
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
			set(origin[Tn >> 5], Tn & 31);
			Xm = Xn;
			operation = Delete;
			printf("accepted\t");
			printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u, Tn: %4u\n", Xi, Xm, Ti, Tm, Tn);
			flag = true;
		}
		if (!flag)printf("denied\n");
	}
	//insert a kink (two neighbour kinks cannot have the same t)
	void insertKinkLimited()
	{
		printf("InsertLimited\t");
		bool flag(false);
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
		unsigned int tb((Tm - tauCD2) & NMinus1), te((Tm + tauCD2M1) & NMinus1);
		kinksLogicOr(Xk, Xk1, Xk2, tb, te);
		unsigned int t0, t1;
		int dd0, dd1;
		unsigned int* origin(gridKinksOrigin(Xk, dim));
		unsigned int kn(kinksNum(tp0, tb, te));
		//kinks number in [Tm - tauCD2, Tm + tauCD2 - 1)
		unsigned int nk(kinksNum(origin, tb, te));
		//make sure that there is space for a new kink
		if (tauCM1 != kn - get(tp0[Tm >> 5], Tm & 31))
			do
			{
				//choose a position which has no kink
				dd0 = rdDeltaC(mt);
				if (dd0 >= 0)
				{
					dd0++;
					Tn = (Tm + dd0) & NMinus1;
					t0 = (Tm + 1) & NMinus1, t1 = Tn;
				}
				else
				{
					Tn = (Tm + dd0) & NMinus1;
					t0 = (Tn + 1) & NMinus1, t1 = Tm;
				}
			} while (get(tp0[Tn >> 5], Tn & 31));
		else
		{
			printf("denied\n");
			return;
		}
		dd1 = dd0;
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
			set(origin[Tn >> 5], Tn & 31);
			gridU[Xm & NMinus1] += dd0;
			gridU[Xn & NMinus1] += dd1;
			Xm = Xn;
			operation = Insert;
			printf("accepted\t");
			printf("Xi:%4d, Xm:%4d, Ti:%4d, Tm:%4u, Tn: %4u\n", Xi, Xm, Ti, Tm, Tn);
			flag = true;
		}
		if (!flag)printf("denied\n");
	}
	//one step of operation
	void operate()
	{
		steps++;
		operation = None;
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
	//wind number
	unsigned int windingNumber()
	{
		return abs(Xi - Xm) / NMinus1;
	}
	//traverse a grid
	void traverse()
	{

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
		printf("Xi:%4d, Xm:%4d, Tm:%4u, Wind:%2u\n", Xi, Xm, Tm, windingNumber());
	}
	//print statistical results
	static void printResults()
	{
		printf("Rounds:%4u, Rm:%4u, R:%.3f, Steps Per Loop:%.3f\n",
			rounds, Rm, float(Rm) / rounds, float(steps) / rounds);
	}
#undef get
#undef set
};

unsigned int Grid::Rm = 0;
unsigned int Grid::rounds = 0;
unsigned int Grid::steps = 0;

int main()
{
	std::mt19937 mt(time(nullptr));
	int s(0);
	for (unsigned int c0(0); c0 < 10000; ++c0)
	{
		int d(rdDeltaC1(mt));
		if (d >= 0)d++;
		printf("%d\n", s += d);
	}
	//Grid grid;

	//Grid::Rm = 0;
	//Grid::rounds = 0;
	//Grid::steps = 0;

	//while (!Grid::rounds)
	//	grid.operate();
	//grid.print();


	/*Timer timer;

	for (unsigned int c0(0); c0 < 100000; ++c0)
		grid.operate();


	timer.begin();
	for (unsigned int c0(0); c0 < 1000000; ++c0)
		grid.operate();
	timer.end();
	timer.print();

	Grid::printResults();*/




	// for (unsigned int c0(0);c0 < 50;++c0)
	// {
	// 	printf("%u:\t", c0);
	// 	for (unsigned int c1(0);c1 < 20000;++c1)
	// 		grid.operate();
	// 	grid.printDebug();
	// }

	// grid.grid[0] = 643786034;
	// grid.grid[1] = 643786034;
	// grid.grid[2] = 643786034;
	// grid.grid[3] = 643786034;
	// grid.print(0);
	// grid.flipSpins(grid.tp0, 0, 77, 5);
	// grid.copySpins(grid.tp0, 0, 77, 5);
	// grid.print(0);

	printf("\n");

	// grid.print();
}
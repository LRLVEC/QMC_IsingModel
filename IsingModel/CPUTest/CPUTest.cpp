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

//dimension: 1+1
//size: N=128
//use int32 instead of int64 because CUDA's support for int64 is simulated


constexpr unsigned int powd(unsigned int a, unsigned int n)
{
	if (n)return powd(a, n - 1) * a;
	else return 1;
}

constexpr unsigned int N(128);
constexpr unsigned int NMinus1(N - 1);
constexpr unsigned int BrickNum(N / 32);
constexpr unsigned int BrickMinus1(BrickNum - 1);
constexpr unsigned int LogN(5);
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

constexpr float h(0.1);
constexpr unsigned int tauA(10);//\tau_a
constexpr unsigned int tauAD2(tauA / 2);//\dfrac{\tau_a}{2}
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
//if return non-negetive, then plus 1 to make sure the result is in [-tauCD2, tauCD2)\{0}
std::uniform_int_distribution<int> rdDelta(-tauCD2, tauCD2 - 2);

//Each Grid for each thread
struct Grid
{
	std::mt19937 mt;

	//use pointer instead of multi-dimension array to simulate CUDA
	unsigned int* grid;
	//stores U
	unsigned int* gridU;
	//stores kink, gridKinks[z][y][x][dim][brick]=gridKinks[brick + BrickNum * (dim + SpaceDim * (x + N * (y + N * z)))],
	unsigned int* gridKinks;
	//stores kink num, gridKinksNum[z][y][x][dim]=gridKinks[dim + SpaceDim * (x + N * (y + N * z))],
	//unsigned int* gridKinksNum;

	int Ti, Tm;//-1 means not created, does not need modulo
	int Xi, Xm;

	unsigned int tp0[BrickNum];
	unsigned int tp1[BrickNum];

#define get(x, y) (((x) >> (y)) & 1)
	// #define set(x) (center ^= (1u << (x)))

	Grid()
		:
		mt(time(nullptr)),
		grid((unsigned int*)malloc(GridSize)),
		gridU((unsigned int*)malloc(GridUSize)),
		gridKinks((unsigned int*)malloc(GridKinksSize))
	{
		clear();
	}
	~Grid()
	{
		free(grid);
		free(gridU);
		free(gridKinks);
	}
	//Clear the grid to original state
	void clear()
	{
		memset(grid, 0, GridSize);
		memset(gridU, 0, GridUSize);
		memset(gridKinks, 0, GridKinksSize);
		Ti = -1;
	}
	inline unsigned int* gridKinksOrigin(unsigned int X, unsigned int dim)
	{
		return gridKinks + BrickNum * (dim + SpaceDim * X);
	}
	//Flips spins in [t0, t1], t1 > t0;
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
			if (Ua == Ub)gridOrigin[Ua] = tp[Ua];
			else
			{
				gridOrigin[Ua] = tp[Ua];
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
		}
		else Ti = -1;
	}
	//annihilate defectes
	void annihilateDefects()
	{
		unsigned int dt(abs(Ti - Tm));
		int dtauM(dt <= N - dt ? dt : N - dt);
		if (dtauM <= tauAD2)
		{
			unsigned int t0, t1;
			if ((Ti > Tm) ^ (dt <= tauAD2))  t0 = Ti, t1 = Tm;
			else t0 = Tm, t1 = Ti;
			int dU(flipSpins(tp0, Xi & NMinus1, t0, t1));
			dtauM -= 2 * dU;//variation of U
			float acceptance(PaTauAInv * expf(h * dtauM));
			if (rd(mt) < acceptance)
			{
				copySpins(tp0, Xi & NMinus1, t0, t1);
				gridU[Xi & NMinus1] += dtauM;
			}
		}
	}
	//move Tm
	void moveMT()
	{

	}
	//insert a kink
	void insertKink()
	{
		int dim(rdWorldLineDim(mt));//which dimension that the movement takes 
		int dir(rdWorldLineDir(mt));//direction of the movement, 0 means -1, 1 means 1
		int Xn(Xm + (2 * dir - 1));//neighbour of Xm, new Xm
		int Xk((dir ? Xm : Xn) & NMinus1);//choose the index of gridKinks
		unsigned int Tn;//position of the add kink
		//make sure that there is space for a new kink
		unsigned int* origin(gridKinksOrigin(Xk, dim));
		unsigned int t0, t1;
		int dd0, dd1;
		//kinks number in [Tm - tauCD2, Tm + tauCD2 - 1)
		unsigned int kn(kinksNum(origin, (Tm - tauCD2) & NMinus1, (Tm + tauCD2M1) & NMinus1));
		if (tauCM1 != kn - get(origin[Tm >> 5], Tm & 31))
			do
			{
				//choose a position which has no kink
				dd0 = rdDelta(mt);
				if (dd0 >= 0)
				{
					dd0++;
					Tn = (Tm + dd0) & NMinus1;
					t0 = Tm, t1 = Tn;
				}
				else
				{
					Tn = (Tm + dd0) & NMinus1;
					t0 = Tn, t1 = Tm;
				}
			} while (get(origin[Tn >> 5], Tn & 31));
		else return;
		dd1 = dd0;
		int dU0(flipSpins(tp0, Xm & NMinus1, (t0 + 1) & NMinus1, t1));
		dd0 -= 2 * dU0;//variation of U
		int dU1(flipSpins(tp1, Xn & NMinus1, (t0 + 1) & NMinus1, t1));
		dd1 -= 2 * dU1;//variation of U
		float acceptance((tauC * expf(h * (dd0 + dd1)) / (kn + 1)));
		if (rd(mt) < acceptance)
		{
			copySpins(tp0, Xm & NMinus1, (t0 + 1) & NMinus1, t1);
			copySpins(tp1, Xn & NMinus1, (t0 + 1) & NMinus1, t1);
			gridU[Xm & NMinus1] += dd0;
			gridU[Xn & NMinus1] += dd1;
			Xm = Xn;
		}
	}
	//delete a kink
	void deleteKink()
	{

	}
	//one step of operation
	void operate()
	{
		if (Ti < 0)
		{
			createDefects();
		}
		else
		{
			float r(rd(mt));
			if (r <= APa && (Xi - Xm) & NMinus1)annihilateDefects();
			else if (r <= APb)
			{
				moveMT();
			}
			else if (r <= APc0)
			{
				insertKink();
			}
			else
			{
				deleteKink();
			}
		}
	}
	//print one world line
	void print(unsigned int X)
	{
		printBit(grid + X * BrickNum, N, 1);
	}
};

int main()
{
	Grid grid;

	grid.grid[0] = 643786034;
	grid.grid[1] = 643786034;
	grid.grid[2] = 643786034;
	grid.grid[3] = 643786034;
	grid.print(0);
	grid.flipSpins(grid.tp0, 0, 5, 7);
	grid.copySpins(grid.tp0, 0, 5, 7);
	grid.print(0);

	int a(-1246);
	printBit(&a, 32, 1);
}
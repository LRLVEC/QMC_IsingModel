#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <_Time.h>
#include <_Bit.h>
#include <random>
#ifdef _WIN32
#include <intrin.h>
#define popc(x) popc(x)
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
constexpr unsigned int TSize(N / 32);
constexpr unsigned int SpaceDimension(1);
constexpr unsigned int GridSize(powd(N, SpaceDimension)* TSize * sizeof(unsigned int));
constexpr unsigned int GridUSize(powd(N, SpaceDimension) * sizeof(unsigned int));

constexpr float Pa(1.f / 3);
constexpr float Pb(1.f / 3);
constexpr float Pc(1.f / 6);
constexpr float APa(Pa);
constexpr float APb(Pa + Pb);
constexpr float APc0(Pa + Pb + Pc);

constexpr float h(0.1);
constexpr unsigned int tauA(10);//\tau_a
constexpr unsigned int tauAD2(tauA / 2);//\dfrac{\tau_a}{2}
constexpr float PaTauA(Pa* tauA);
constexpr float PaTauAInv(1 / PaTauA);


int Ti, Tm;//-1 means not created, does not need modulo
int Xi, Xm;
std::mt19937 mt(time(nullptr));
std::uniform_real_distribution<float> rd(0, 1);
std::uniform_int_distribution<unsigned int> rdint(0, N - 1);
std::uniform_int_distribution<unsigned int> rdtauA(1, tauA);

void ClearGrid(unsigned int* grid, unsigned int* gridU)
{
	memset(grid, 0, GridSize);
	memset(gridU, 0, GridUSize);
	Ti = -1;
}
//Flips spins in [t0, t1], t1 > t0;
//make sure that 0 <= t0 < N, 0 <= t1 < N;
//stores the result in tp, return the original U of flipped spins
unsigned int FlipSpins(unsigned int* grid, unsigned int* tp, unsigned int X, unsigned int t0, unsigned int t1)
{
	unsigned int* gridOrigin(grid + TSize * X);
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
			mask >>= popc(31 - Db + Da);
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
unsigned int SpinUpNum(unsigned int* grid, unsigned int X)
{
	unsigned int a(0);
	unsigned int* gridOrigin(grid + TSize * X);
	for (unsigned int c0(0); c0 < BrickNum; ++c0)
		a += popc(gridOrigin[c0]);
	return a;
}
void CopySpins(unsigned int* grid, unsigned int* tp, unsigned int X, unsigned int t0, unsigned int t1)
{
	unsigned int* gridOrigin(grid + TSize * X);
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
void CreateDefects(unsigned int* grid, unsigned int* gridU)
{
	static unsigned int tp[BrickNum];
	Xi = Xm = rdint(mt);
	Ti = rdint(mt);
	int dtauM(rdtauA(mt));
	Tm = (Ti + dtauM) & NMinus1;
	//Flip spins in (Ti, Tm]
	int dU(FlipSpins(grid, tp, Xi, Ti + 1, Tm));
	dtauM -= 2 * dU;
	float acceptance(PaTauA * expf(h * dtauM));
	if (rd(mt) < acceptance)
	{
		CopySpins(grid, tp, Xi, Ti + 1, Tm);
		gridU[Xi] += dtauM;
	}
	else Ti = -1;
}
void AnnihilateDefects(unsigned int* grid, unsigned int* gridU)
{
	static unsigned int tp[BrickNum];
	unsigned int dt(abs(Ti - Tm));
	int dtauM(dt <= N - dt ? dt : N - dt);
	if (dtauM <= tauAD2)
	{
		unsigned int t0, t1;
		if ((Ti > Tm) ^ (dt <= tauAD2))  t0 = Ti, t1 = Tm;
		else t0 = Tm, t1 = Ti;
		int dU(FlipSpins(grid, tp, Xi, t0, t1));
		dtauM -= 2 * dU;
		float acceptance(PaTauAInv * expf(h * dtauM));
		if (rd(mt) < acceptance)
		{
			CopySpins(grid, tp, Xi, t0, t1);
			gridU[Xi] += dtauM;
		}
	}
}
void MoveMT(unsigned int* grid)
{

}
void InsertKink(unsigned int* grid)
{

}
void DeleteKink(unsigned int* grid)
{

}
void Operate(unsigned int* grid, unsigned int* gridU)
{
	if (Ti < 0)
	{
		CreateDefects(grid, gridU);
	}
	else
	{
		float r(rd(mt));
		if (r <= APa && Xi == Xm)AnnihilateDefects(grid, gridU);
		else if (r <= APb)
		{

		}
		else if (r <= APc0)
		{

		}
		else
		{

		}
	}
}

int main()
{
	unsigned int* grid((unsigned int*)malloc(GridSize));//use pointer instead of multi-dimension array to simulate CUDA
	unsigned int* gridU((unsigned int*)malloc(GridUSize));//stores U
	ClearGrid(grid, gridU);

	grid[0] = 6437869034;
	grid[1] = 6437869034;
	grid[2] = 6437869034;
	grid[3] = 6437869034;
	unsigned int tp[BrickNum];
	printBit(grid, N, 1);
	FlipSpins(grid, tp, 0, 7, 5);
	CopySpins(grid, tp, 0, 7, 5);
	printBit(grid, N, 1);

	free(grid);
	free(gridU);
}
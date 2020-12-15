#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <_Time.h>
#include <_Bit.h>
#include <random>
#include <intrin.h>

//dimension: 1+1
//size: N=128
//use int32 instead of int64 because CUDA's support for int64 is simulated

constexpr unsigned int powd(unsigned int a, unsigned int n)
{
	if (n)return powd(a, n - 1) * a;
	else return 1;
}

constexpr unsigned int N(128);
constexpr unsigned int TSize(N / 32);
constexpr unsigned int SpaceDimension(1);
constexpr unsigned int GridSize(powd(N, SpaceDimension)* TSize * sizeof(unsigned int));

constexpr float Pa(1.f / 3);
constexpr float Pb(1.f / 3);
constexpr float Pc(1.f / 6);
constexpr float APa(Pa);
constexpr float APb(Pa + Pb);
constexpr float APc0(Pa + Pb + Pc);

constexpr unsigned int tauA(10);//\tau_a

int Ti, Tm;//-1 means not created
int Xi, Xm;
std::mt19937 mt(time(nullptr));
std::uniform_real_distribution<float> rd(0, 1);
std::uniform_int_distribution<unsigned int> rdint(0, N - 1);
std::uniform_int_distribution<unsigned int> rdtauA(1, tauA);

void ClearGrid(unsigned int* grid)
{
	memset(grid, 0, GridSize);
}
void FlipSpins(unsigned int* grid, unsigned int X, unsigned int t0, unsigned int t1)
{

}
void CreateDefects(unsigned int* grid)
{
	Xi = Xm = rdint(mt);
	Ti = rdint(mt);
	Tm = Ti + rdtauA(mt);

	//__popcnt();
}
void AnnihilateDefects(unsigned int* grid)
{

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
void Operate(unsigned int* grid)
{
	if (Ti < 0)
	{
		CreateDefects(grid);
	}
	else
	{
		float r(rd(mt));
		if (r <= APa)AnnihilateDefects(grid);
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
	ClearGrid(grid);

	grid[0] = 1024;

	printBit(grid, 128, 1);

	free(grid);
}
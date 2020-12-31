#version 450 core
#define N 64
#define NMinus1 63
#define LogN 6
#define BrickNum 2
#define get(x, y) (((x) >> (y)) & 1)
//layout(std140, row_major, binding = 0)uniform transBuffer
//{
//	mat4 trans;
//};
layout(location = 0)in vec2 position;
layout(std430, binding = 1)buffer SpinsBuffer
{
	unsigned int spins[];
};
layout(std430, binding = 2)buffer SpinKinksBuffer
{
	unsigned int spinKinks[];
};
//layout(location = 1)in vec3 velocity;
out vec4 fragColor;
void main()
{
	gl_Position = vec4(position, 0, 1);
	unsigned int id = gl_VertexID >> 1;
	unsigned int T = id & NMinus1;
	unsigned int X = id >> LogN;
	unsigned int offset = X * BrickNum;
	if (get(spinKinks[offset + (T >> 5)], T & 31) == 1)
		fragColor = vec4(0.862f, 0.f, 0.f, 1.f);
	else
		fragColor = vec4(0.f);
}
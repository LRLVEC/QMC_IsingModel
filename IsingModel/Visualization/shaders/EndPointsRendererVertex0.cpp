#version 450 core
layout(location = 0)in vec2 position;
out vec4 fragColor;
void main()
{
	gl_Position = vec4(position, 0, 1);
	if (gl_VertexID == 0)
		fragColor = vec4(1.f, 0.725f, 0.f, 1.f);
	else
		fragColor = vec4(0.f, 0.725f, 1.f, 1.f);
}
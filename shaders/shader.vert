#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 out_color;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
	mat4 projection;
} global_uniforms;

layout(std140, set = 1, binding = 0) uniform ObjectUniforms {
	mat4 model;
	vec4 tint;
} object_uniforms;

void main() {
	gl_Position = global_uniforms.projection * object_uniforms.model * vec4(in_position, 1.0);
	out_color = in_color * object_uniforms.tint.rgb;
}

#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_normal;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
	mat4 view_projection;
	vec3 camera_position;
	vec3 directional_direction;
	vec3 directional_color;
	vec3 ambient_light;
	uint point_light_count;
	uint spot_light_count;
	float marker_radius;
	float cone_length;
} global_uniforms;

layout(std140, set = 1, binding = 0) uniform ObjectUniforms {
	mat4 model;
	vec3 albedo;
	vec3 specular;
	float shininess;
} object_uniforms;

void main() {
	vec4 world_position = object_uniforms.model * vec4(in_position, 1.0);

	out_world_position = world_position.xyz;
	out_normal = transpose(inverse(mat3(object_uniforms.model))) * in_normal;

	gl_Position = global_uniforms.view_projection * world_position;
}

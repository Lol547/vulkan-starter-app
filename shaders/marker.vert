#version 450 core

layout(location = 0) in vec3 in_position;

layout(location = 0) out vec3 out_color;

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

struct PointLight {
	vec3 position;
	float intensity;
	vec3 color;
	float _padding;
};

struct SpotLight {
	vec3 position;
	float intensity;
	vec3 color;
	float cos_inner;
	vec3 direction;
	float cos_outer;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

void main() {
	uint index = uint(gl_InstanceIndex);

	vec3 center;
	vec3 color;
	if (index < global_uniforms.point_light_count) {
		PointLight light = point_lights[index];
		center = light.position;
		color = light.color;
	} else {
		SpotLight light = spot_lights[index - global_uniforms.point_light_count];
		center = light.position;
		color = light.color;
	}

	gl_Position = global_uniforms.view_projection * vec4(center + in_position * global_uniforms.marker_radius, 1.0);
	out_color = color;
}

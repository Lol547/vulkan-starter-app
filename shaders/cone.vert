#version 450 core

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

struct SpotLight {
	vec3 position;
	float intensity;
	vec3 color;
	float cos_inner;
	vec3 direction;
	float cos_outer;
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

const int SEGMENTS = 24;
const int RIB_STEP = 6;
const float TWO_PI = 6.28318530718;

void main() {
	SpotLight light = spot_lights[gl_InstanceIndex];

	vec3 axis = normalize(light.direction);
	vec3 helper = (abs(axis.y) < 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent = normalize(cross(helper, axis));
	vec3 bitangent = cross(axis, tangent);

	float c = clamp(light.cos_outer, 0.01, 0.9999);
	float radius = global_uniforms.cone_length * sqrt(1.0 - c * c) / c;
	vec3 base_center = light.position + axis * global_uniforms.cone_length;

	int v = gl_VertexIndex;
	int ring_vertices = SEGMENTS * 2;

	vec3 world_position;
	if (v < ring_vertices) {
		int point_index = v / 2 + (v % 2);
		float angle = TWO_PI * float(point_index) / float(SEGMENTS);
		world_position = base_center + radius * (cos(angle) * tangent + sin(angle) * bitangent);
	} else {
		int r = v - ring_vertices;
		if (r % 2 == 0) {
			world_position = light.position;
		} else {
			float angle = TWO_PI * float((r / 2) * RIB_STEP) / float(SEGMENTS);
			world_position = base_center + radius * (cos(angle) * tangent + sin(angle) * bitangent);
		}
	}

	gl_Position = global_uniforms.view_projection * vec4(world_position, 1.0);
	out_color = vec3(1.0);
}

#version 450 core

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec4 out_color;

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

vec3 blinn_phong(vec3 n, vec3 v, vec3 l, vec3 radiance, vec3 kd, vec3 ks, float shininess) {
	float cos_theta = dot(n, l);
	if (cos_theta <= 0.0) {
		return vec3(0.0);
	}

	vec3 h = normalize(l + v);
	float cos_phi = max(dot(n, h), 0.0);
	return radiance * (cos_theta * kd + pow(cos_phi, shininess) * ks);
}

void main() {
	vec3 n = normalize(in_normal);
	vec3 v = normalize(global_uniforms.camera_position - in_world_position);

	vec3 kd = object_uniforms.albedo;
	vec3 ks = object_uniforms.specular;

	vec3 color = global_uniforms.ambient_light * kd;

	vec3 l = -normalize(global_uniforms.directional_direction);
	color += blinn_phong(n, v, l, global_uniforms.directional_color, kd, ks, object_uniforms.shininess);

	uint point_count = min(global_uniforms.point_light_count, uint(point_lights.length()));
	for (uint i = 0; i < point_count; ++i) {
		PointLight light = point_lights[i];

		vec3 to_light = light.position - in_world_position;
		float dist2 = max(dot(to_light, to_light), 1e-4);
		vec3 l = to_light * inversesqrt(dist2);

		vec3 radiance = light.color * light.intensity / dist2;
		color += blinn_phong(n, v, l, radiance, kd, ks, object_uniforms.shininess);
	}

	uint spot_count = min(global_uniforms.spot_light_count, uint(spot_lights.length()));
	for (uint i = 0; i < spot_count; ++i) {
		SpotLight light = spot_lights[i];

		vec3 to_light = light.position - in_world_position;
		float dist2 = max(dot(to_light, to_light), 1e-4);
		vec3 l = to_light * inversesqrt(dist2);

		float cos_angle = dot(-l, normalize(light.direction));
		float t = clamp((cos_angle - light.cos_outer) / max(light.cos_inner - light.cos_outer, 1e-4), 0.0, 1.0);
		float cone = t * t * (3.0 - 2.0 * t);

		vec3 radiance = light.color * light.intensity * cone / dist2;
		color += blinn_phong(n, v, l, radiance, kd, ks, object_uniforms.shininess);
	}

	out_color = vec4(color, 1.0);
}

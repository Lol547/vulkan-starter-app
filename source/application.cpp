#include "application.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>

#include <imgui.h>

namespace application {

namespace {

constexpr float pi = 3.14159265358979323846f;

struct Mat4 {
	float m[16];
};

Mat4 identity() {
	Mat4 r{};
	r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
	return r;
}

Mat4 mul(const Mat4& a, const Mat4& b) {
	Mat4 r{};
	for (int c = 0; c < 4; ++c) {
		for (int row = 0; row < 4; ++row) {
			float sum = 0.0f;
			for (int k = 0; k < 4; ++k) {
				sum += a.m[k * 4 + row] * b.m[c * 4 + k];
			}
			r.m[c * 4 + row] = sum;
		}
	}
	return r;
}

Mat4 translate(float x, float y, float z) {
	Mat4 r = identity();
	r.m[12] = x;
	r.m[13] = y;
	r.m[14] = z;
	return r;
}

Mat4 scale(float x, float y, float z) {
	Mat4 r = identity();
	r.m[0] = x;
	r.m[5] = y;
	r.m[10] = z;
	return r;
}

Mat4 rotateX(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[5] = c;  r.m[9] = -s;
	r.m[6] = s;  r.m[10] = c;
	return r;
}

Mat4 rotateY(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[0] = c;  r.m[8] = s;
	r.m[2] = -s; r.m[10] = c;
	return r;
}

Mat4 rotateZ(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[0] = c;  r.m[4] = -s;
	r.m[1] = s;  r.m[5] = c;
	return r;
}

Mat4 perspective(float fov_y, float aspect, float z_near, float z_far) {
	const float t = 1.0f / std::tan(fov_y * 0.5f);
	Mat4 r{};
	r.m[0] = t / aspect;
	r.m[5] = t;
	r.m[10] = z_far / (z_far - z_near);
	r.m[11] = 1.0f;
	r.m[14] = -z_near * z_far / (z_far - z_near);
	return r;
}

Mat4 orthographic(float half_w, float half_h, float z_near, float z_far) {
	Mat4 r{};
	r.m[0] = 1.0f / half_w;
	r.m[5] = 1.0f / half_h;
	r.m[10] = 1.0f / (z_far - z_near);
	r.m[14] = -z_near / (z_far - z_near);
	r.m[15] = 1.0f;
	return r;
}

float radians(float degrees) {
	return degrees * pi / 180.0f;
}

struct Vertex {
	float position[3];
	float normal[3];
};

struct GlobalUniforms {
	float viewProjection[16];
	alignas(16) float cameraPosition[3];
	alignas(16) float directionalDirection[3];
	alignas(16) float directionalColor[3];
	alignas(16) float ambientLight[3];
	uint32_t pointLightCount;
	uint32_t spotLightCount;
	float markerRadius;
	float coneLength;
};

constexpr uint32_t max_point_lights = 64;
constexpr uint32_t max_spot_lights = 64;

struct GpuPointLight {
	float position[3];
	float intensity;
	float color[3];
	float _padding;
};

struct GpuSpotLight {
	float position[3];
	float intensity;
	float color[3];
	float cosInner;
	float direction[3];
	float cosOuter;
};

static_assert(sizeof(GpuPointLight) == 32, "GpuPointLight must match std430 layout");
static_assert(sizeof(GpuSpotLight) == 48, "GpuSpotLight must match std430 layout");

constexpr VkDeviceSize point_light_buffer_size = max_point_lights * sizeof(GpuPointLight);
constexpr VkDeviceSize spot_light_buffer_size = max_spot_lights * sizeof(GpuSpotLight);

struct PointLight {
	float position[3];
	float color[3];
	float intensity;
};

struct SpotLight {
	float position[3];
	float direction[3];
	float color[3];
	float intensity;
	float inner_angle;
	float outer_angle;
};

struct ObjectUniforms {
	float model[16];
	alignas(16) float albedo[3];
	alignas(16) float specular[3];
	float shininess;
};

constexpr uint32_t max_objects = 4;

struct SceneObject {
	float position[3];
	float rotation[3];
	float scale[3];
	float albedo[3];
	float specular[3];
	float shininess;
};

#ifndef SHADER_DIR
#define SHADER_DIR "shaders/"
#endif

constexpr char vertex_shader_path[] = SHADER_DIR "shader.vert.spv";
constexpr char fragment_shader_path[] = SHADER_DIR "shader.frag.spv";
constexpr char marker_vertex_shader_path[] = SHADER_DIR "marker.vert.spv";
constexpr char marker_fragment_shader_path[] = SHADER_DIR "marker.frag.spv";
constexpr char cone_vertex_shader_path[] = SHADER_DIR "cone.vert.spv";

constexpr uint32_t cone_vertex_count = 24 * 2 + 4 * 2;

constexpr uint32_t sphere_rings = 100;
constexpr uint32_t sphere_slices = 100;

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};

Buffer vertex_buffer;
Buffer index_buffer;
uint32_t index_count = 0;

Buffer global_uniform_buffer;
Buffer point_light_buffer;
Buffer spot_light_buffer;
VkDescriptorSetLayout global_set_layout = VK_NULL_HANDLE;
VkDescriptorSet global_descriptor_set = VK_NULL_HANDLE;

Buffer object_uniform_buffers[max_objects];
VkDescriptorSetLayout object_set_layout = VK_NULL_HANDLE;
VkDescriptorSet object_descriptor_sets[max_objects] = {};

VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkPipeline marker_pipeline = VK_NULL_HANDLE;
VkPipeline cone_pipeline = VK_NULL_HANDLE;

SceneObject objects[max_objects];
int object_count = 3;
int selected_object = 0;

int projection_mode = 0;
float fov_degrees = 60.0f;
float ortho_size = 5.0f;
float near_plane = 0.1f;
float far_plane = 100.0f;

float camera_position[3] = { 0.0f, 0.0f, 0.0f };
float camera_rotation[3] = { 0.0f, 0.0f, 0.0f };

float dir_light_direction[3] = { 0.4f, 0.5f, 1.0f };
float dir_light_color[3] = { 1.0f, 1.0f, 1.0f };
float dir_light_intensity = 1.0f;
float ambient_light[3] = { 0.1f, 0.1f, 0.1f };

std::vector<PointLight> point_lights;
std::vector<SpotLight> spot_lights;

bool show_light_markers = true;
bool show_spot_cones = true;
float marker_radius = 0.1f;
float cone_length = 1.5f;

bool anim_playing = true;
float anim_speed = 1.0f;
float anim_radius = 0.6f;
float anim_frequency = 1.0f;
float anim_spin_speed = 45.0f;
double anim_time = 0.0;
double last_time = -1.0;

void resetObject(uint32_t i) {
	static const float positions[max_objects][3] = {
		{ 0.0f, 0.0f, 7.0f },
		{ -2.5f, 0.0f, 7.0f },
		{ 2.5f, 0.0f, 7.0f },
		{ 0.0f, 2.2f, 7.0f },
	};
	static const float scales[max_objects] = { 1.0f, 0.6f, 0.6f, 0.5f };
	static const float albedos[max_objects][3] = {
		{ 0.9f, 0.3f, 0.3f },
		{ 0.3f, 0.8f, 0.4f },
		{ 0.3f, 0.5f, 0.9f },
		{ 0.9f, 0.8f, 0.3f },
	};

	SceneObject& o = objects[i];
	std::memcpy(o.position, positions[i], sizeof(o.position));
	o.rotation[0] = o.rotation[1] = o.rotation[2] = 0.0f;
	o.scale[0] = o.scale[1] = o.scale[2] = scales[i];
	std::memcpy(o.albedo, albedos[i], sizeof(o.albedo));
	o.specular[0] = o.specular[1] = o.specular[2] = 0.5f;
	o.shininess = 32.0f;
}

void generateSphere(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
	const auto makeVertex = [](float x, float y, float z) {
		return Vertex{ { x, y, z }, { x, y, z } };
	};

	vertices.push_back(makeVertex(0.0f, 1.0f, 0.0f));
	vertices.push_back(makeVertex(0.0f, -1.0f, 0.0f));

	for (uint32_t r = 0; r < sphere_rings; ++r) {
		const float phi = pi * float(r + 1) / float(sphere_rings + 1);
		for (uint32_t s = 0; s < sphere_slices; ++s) {
			const float theta = 2.0f * pi * float(s) / float(sphere_slices);
			vertices.push_back(makeVertex(std::sin(phi) * std::cos(theta),
			                              std::cos(phi),
			                              std::sin(phi) * std::sin(theta)));
		}
	}

	const auto ring = [](uint32_t r, uint32_t s) {
		return 2 + r * sphere_slices + (s % sphere_slices);
	};

	for (uint32_t s = 0; s < sphere_slices; ++s) {
		indices.insert(indices.end(), { 0, ring(0, s + 1), ring(0, s) });
	}

	for (uint32_t r = 0; r + 1 < sphere_rings; ++r) {
		for (uint32_t s = 0; s < sphere_slices; ++s) {
			const uint32_t a = ring(r, s), b = ring(r, s + 1);
			const uint32_t c = ring(r + 1, s), d = ring(r + 1, s + 1);
			indices.insert(indices.end(), { a, b, c, b, d, c });
		}
	}

	for (uint32_t s = 0; s < sphere_slices; ++s) {
		indices.insert(indices.end(), { 1, ring(sphere_rings - 1, s), ring(sphere_rings - 1, s + 1) });
	}
}


bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	auto& context = graphics::internal::context;

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
		         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo result_info{};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &out.buffer, &out.allocation, &result_info) != VK_SUCCESS) {
		std::cerr << "Failed to create and allocate buffer\n";
		return false;
	}

	out.mapped = result_info.pMappedData;
	return true;
}

void uploadToBuffer(const Buffer& buffer, const void* data, size_t size) {
	auto& context = graphics::internal::context;

	std::memcpy(buffer.mapped, data, size);
	vmaFlushAllocation(context.allocator, buffer.allocation, 0, VK_WHOLE_SIZE);
}

void destroyBuffer(Buffer& buffer) {
	auto& context = graphics::internal::context;

	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

VkShaderModule loadShaderModule(const char* path) {
	auto& context = graphics::internal::context;

	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << path << '\n';
		return VK_NULL_HANDLE;
	}

	const size_t size = size_t(file.tellg());
	if (size == 0 || size % sizeof(uint32_t) != 0) {
		std::cerr << "Invalid SPIR-V file size: " << path << '\n';
		return VK_NULL_HANDLE;
	}

	std::vector<uint32_t> code(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));
	file.close();

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code.data(),
	};

	VkShaderModule result = VK_NULL_HANDLE;
	if (vkCreateShaderModule(context.device, &info, nullptr, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << path << '\n';
		return VK_NULL_HANDLE;
	}

	return result;
}

bool createPipeline(const char* vertex_path, const char* fragment_path,
                    VkPrimitiveTopology topology, bool use_vertex_input,
                    VkPipeline& out_pipeline) {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = loadShaderModule(vertex_path);
	VkShaderModule fragment_shader = loadShaderModule(fragment_path);

	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stage_infos[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_bindings[] = {
		{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		},
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = sizeof(vertex_bindings) / sizeof(vertex_bindings[0]),
		.pVertexBindingDescriptions = vertex_bindings,
		.vertexAttributeDescriptionCount = sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineVertexInputStateCreateInfo empty_vertex_input_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	};

	const VkPipelineInputAssemblyStateCreateInfo assembly_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo raster_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo sample_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo blend_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stage_infos) / sizeof(stage_infos[0]),
		.pStages = stage_infos,
		.pVertexInputState = use_vertex_input ? &vertex_input_state : &empty_vertex_input_state,
		.pInputAssemblyState = &assembly_state,
		.pViewportState = &viewport_state,
		.pRasterizationState = &raster_state,
		.pMultisampleState = &sample_state,
		.pDepthStencilState = &depth_state,
		.pColorBlendState = &blend_state,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &out_pipeline);

	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

bool createPipelines() {
	return
		createPipeline(vertex_shader_path, fragment_shader_path,
		               VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true, pipeline) &&
		createPipeline(marker_vertex_shader_path, marker_fragment_shader_path,
		               VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true, marker_pipeline) &&
		createPipeline(cone_vertex_shader_path, marker_fragment_shader_path,
		               VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false, cone_pipeline);
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	const VkShaderStageFlags both_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	const VkDescriptorSetLayoutBinding global_bindings[] = {
		{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = both_stages },
		{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = both_stages },
		{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = both_stages },
	};

	const VkDescriptorSetLayoutBinding object_bindings[] = {
		{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = both_stages },
	};

	const VkDescriptorSetLayoutCreateInfo global_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(global_bindings) / sizeof(global_bindings[0]),
		.pBindings = global_bindings,
	};

	const VkDescriptorSetLayoutCreateInfo object_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(object_bindings) / sizeof(object_bindings[0]),
		.pBindings = object_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &global_layout_info, nullptr,
	                                &global_set_layout) != VK_SUCCESS ||
	    vkCreateDescriptorSetLayout(context.device, &object_layout_info, nullptr,
	                                &object_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layouts\n";
		return false;
	}

	const VkDescriptorSetLayout pipeline_set_layouts[] = { global_set_layout, object_set_layout };

	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(pipeline_set_layouts) / sizeof(pipeline_set_layouts[0]),
		.pSetLayouts = pipeline_set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr,
	                           &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1 + max_objects,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 2,
		},
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + max_objects,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo global_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &global_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &global_allocate_info,
	                             &global_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate global descriptor set\n";
		return false;
	}

	VkDescriptorSetLayout object_layouts[max_objects];
	for (uint32_t i = 0; i < max_objects; ++i) {
		object_layouts[i] = object_set_layout;
	}

	const VkDescriptorSetAllocateInfo object_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = max_objects,
		.pSetLayouts = object_layouts,
	};

	if (vkAllocateDescriptorSets(context.device, &object_allocate_info,
	                             object_descriptor_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate object descriptor sets\n";
		return false;
	}

	constexpr uint32_t global_binding_count = 3;
	constexpr uint32_t write_count = global_binding_count + max_objects;

	VkDescriptorBufferInfo buffer_infos[write_count];
	VkWriteDescriptorSet writes[write_count];

	const VkBuffer global_buffers[global_binding_count] = {
		global_uniform_buffer.buffer,
		point_light_buffer.buffer,
		spot_light_buffer.buffer,
	};
	const VkDeviceSize global_ranges[global_binding_count] = {
		sizeof(GlobalUniforms),
		point_light_buffer_size,
		spot_light_buffer_size,
	};
	const VkDescriptorType global_types[global_binding_count] = {
		VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	};

	for (uint32_t b = 0; b < global_binding_count; ++b) {
		buffer_infos[b] = {
			.buffer = global_buffers[b],
			.offset = 0,
			.range = global_ranges[b],
		};

		writes[b] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = global_descriptor_set,
			.dstBinding = b,
			.descriptorCount = 1,
			.descriptorType = global_types[b],
			.pBufferInfo = &buffer_infos[b],
		};
	}

	for (uint32_t i = 0; i < max_objects; ++i) {
		const uint32_t w = global_binding_count + i;

		buffer_infos[w] = {
			.buffer = object_uniform_buffers[i].buffer,
			.offset = 0,
			.range = sizeof(ObjectUniforms),
		};

		writes[w] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = object_descriptor_sets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[w],
		};
	}

	vkUpdateDescriptorSets(context.device, write_count, writes, 0, nullptr);

	return true;
}

void destroyResources() {
	auto& context = graphics::internal::context;

	vkDestroyPipeline(context.device, cone_pipeline, nullptr);
	vkDestroyPipeline(context.device, marker_pipeline, nullptr);
	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, global_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, object_set_layout, nullptr);

	for (uint32_t i = 0; i < max_objects; ++i) {
		destroyBuffer(object_uniform_buffers[i]);
	}
	destroyBuffer(spot_light_buffer);
	destroyBuffer(point_light_buffer);
	destroyBuffer(global_uniform_buffer);
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);

	pipeline = VK_NULL_HANDLE;
	marker_pipeline = VK_NULL_HANDLE;
	cone_pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE;
	global_set_layout = VK_NULL_HANDLE;
	object_set_layout = VK_NULL_HANDLE;
}

Mat4 computeModelMatrix(uint32_t i) {
	const SceneObject& o = objects[i];

	float px = o.position[0], py = o.position[1], pz = o.position[2];
	float rx = o.rotation[0], ry = o.rotation[1], rz = o.rotation[2];

	const float phase = 2.0f * pi * float(i) / float(max_objects);
	const float t = float(anim_time) * anim_frequency;

	px += anim_radius * std::cos(t + phase);
	py += anim_radius * std::sin(2.0f * t + phase);
	pz += 0.5f * anim_radius * std::cos(3.0f * t + phase);

	const float spin = anim_spin_speed * float(anim_time);
	ry += spin;
	rx += 0.5f * spin;

	Mat4 model = scale(o.scale[0], o.scale[1], o.scale[2]);
	model = mul(rotateX(radians(rx)), model);
	model = mul(rotateY(radians(ry)), model);
	model = mul(rotateZ(radians(rz)), model);
	model = mul(translate(px, py, pz), model);
	return model;
}

void normalize3(const float in[3], float out[3]) {
	const float len = std::sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2]);
	if (len > 1e-6f) {
		out[0] = in[0] / len;
		out[1] = in[1] / len;
		out[2] = in[2] / len;
	} else {
		out[0] = 0.0f;
		out[1] = 1.0f;
		out[2] = 0.0f;
	}
}

Mat4 computeViewMatrix() {
	Mat4 view = translate(-camera_position[0], -camera_position[1], -camera_position[2]);
	view = mul(rotateZ(radians(-camera_rotation[2])), view);
	view = mul(rotateY(radians(-camera_rotation[1])), view);
	view = mul(rotateX(radians(-camera_rotation[0])), view);
	return view;
}

}

bool initialize() {
	for (uint32_t i = 0; i < max_objects; ++i) {
		resetObject(i);
	}

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	generateSphere(vertices, indices);
	index_count = uint32_t(indices.size());

	const size_t vertices_size = vertices.size() * sizeof(Vertex);
	const size_t indices_size = indices.size() * sizeof(uint32_t);

	if (!createBuffer(vertices_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
	    !createBuffer(indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) {
		destroyResources();
		return false;
	}

	uploadToBuffer(vertex_buffer, vertices.data(), vertices_size);
	uploadToBuffer(index_buffer, indices.data(), indices_size);

	if (!createBuffer((sizeof(GlobalUniforms) + 0xf) & ~size_t(0xf),
	                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, global_uniform_buffer)) {
		destroyResources();
		return false;
	}

	if (!createBuffer(point_light_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, point_light_buffer) ||
	    !createBuffer(spot_light_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spot_light_buffer)) {
		destroyResources();
		return false;
	}

	for (uint32_t i = 0; i < max_objects; ++i) {
		if (!createBuffer((sizeof(ObjectUniforms) + 0xf) & ~size_t(0xf),
		                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, object_uniform_buffers[i])) {
			destroyResources();
			return false;
		}
	}

	if (!createDescriptors() || !createPipelines()) {
		destroyResources();
		return false;
	}

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	destroyResources();
}

void update(double time) {
	const double dt = (last_time < 0.0) ? 0.0 : (time - last_time);
	last_time = time;
	if (anim_playing) {
		anim_time += dt * double(anim_speed);
	}

	ImGui::Begin("Projection");

	ImGui::RadioButton("Perspective", &projection_mode, 0);
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", &projection_mode, 1);

	if (projection_mode == 0) {
		ImGui::SliderFloat("FOV (deg)", &fov_degrees, 10.0f, 120.0f);
	} else {
		ImGui::SliderFloat("Ortho size", &ortho_size, 1.0f, 15.0f);
	}
	ImGui::SliderFloat("Near", &near_plane, 0.01f, 5.0f);
	ImGui::SliderFloat("Far", &far_plane, 6.0f, 200.0f);

	ImGui::End();

	ImGui::Begin("Camera");

	ImGui::DragFloat3("Position", camera_position, 0.02f);
	ImGui::DragFloat3("Rotation (deg)", camera_rotation, 0.5f);
	if (ImGui::Button("Reset camera")) {
		for (int i = 0; i < 3; ++i) {
			camera_position[i] = 0.0f;
			camera_rotation[i] = 0.0f;
		}
	}

	ImGui::End();

	ImGui::Begin("Lighting");

	ImGui::SeparatorText("Directional light");
	ImGui::DragFloat3("Direction", dir_light_direction, 0.01f);
	ImGui::ColorEdit3("Color", dir_light_color);
	ImGui::SliderFloat("Intensity", &dir_light_intensity, 0.0f, 5.0f);

	ImGui::SeparatorText("Ambient");
	ImGui::ColorEdit3("Ambient light", ambient_light);

	ImGui::SeparatorText("Light visualization");
	ImGui::Checkbox("Show light markers", &show_light_markers);
	ImGui::Checkbox("Show spot cones", &show_spot_cones);
	ImGui::SliderFloat("Marker radius", &marker_radius, 0.02f, 0.5f);
	ImGui::SliderFloat("Cone length", &cone_length, 0.2f, 5.0f);

	ImGui::End();

	ImGui::Begin("Point lights");

	ImGui::Text("%d / %d", int(point_lights.size()), int(max_point_lights));
	if (ImGui::Button("Add point light") && point_lights.size() < max_point_lights) {
		point_lights.push_back(PointLight{ { 0.0f, -2.0f, 4.0f }, { 1.0f, 1.0f, 1.0f }, 30.0f });
	}

	int remove_point = -1;
	for (int i = 0; i < int(point_lights.size()); ++i) {
		PointLight& light = point_lights[size_t(i)];

		ImGui::PushID(i);
		if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Point light %d", i)) {
			ImGui::DragFloat3("Position", light.position, 0.02f);
			ImGui::ColorEdit3("Color", light.color);
			ImGui::DragFloat("Intensity", &light.intensity, 0.5f, 0.0f, 1000.0f);
			if (ImGui::Button("Remove")) {
				remove_point = i;
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}
	if (remove_point >= 0) {
		point_lights.erase(point_lights.begin() + remove_point);
	}

	ImGui::End();

	ImGui::Begin("Spot lights");

	ImGui::Text("%d / %d", int(spot_lights.size()), int(max_spot_lights));
	if (ImGui::Button("Add spot light") && spot_lights.size() < max_spot_lights) {
		spot_lights.push_back(SpotLight{ { 0.0f, -3.0f, 3.0f }, { 0.0f, 0.6f, 0.8f },
		                                 { 1.0f, 1.0f, 1.0f }, 60.0f, 15.0f, 25.0f });
	}

	int remove_spot = -1;
	for (int i = 0; i < int(spot_lights.size()); ++i) {
		SpotLight& light = spot_lights[size_t(i)];

		ImGui::PushID(i);
		if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Spot light %d", i)) {
			ImGui::DragFloat3("Position", light.position, 0.02f);
			ImGui::DragFloat3("Direction", light.direction, 0.01f);
			ImGui::ColorEdit3("Color", light.color);
			ImGui::DragFloat("Intensity", &light.intensity, 0.5f, 0.0f, 1000.0f);
			ImGui::SliderFloat("Outer angle (deg)", &light.outer_angle, 1.0f, 89.0f);
			if (light.inner_angle > light.outer_angle) {
				light.inner_angle = light.outer_angle;
			}
			ImGui::SliderFloat("Inner angle (deg)", &light.inner_angle, 0.0f, light.outer_angle);
			if (ImGui::Button("Remove")) {
				remove_spot = i;
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}
	if (remove_spot >= 0) {
		spot_lights.erase(spot_lights.begin() + remove_spot);
	}

	ImGui::End();

	ImGui::Begin("Objects");

	ImGui::SliderInt("Object count", &object_count, 1, int(max_objects));
	if (selected_object >= object_count) {
		selected_object = object_count - 1;
	}
	ImGui::SliderInt("Selected object", &selected_object, 0, object_count - 1);

	SceneObject& o = objects[selected_object];

	ImGui::SeparatorText("Transform");
	ImGui::DragFloat3("Position", o.position, 0.02f);
	ImGui::DragFloat3("Rotation (deg)", o.rotation, 0.5f);
	ImGui::DragFloat3("Scale", o.scale, 0.01f, 0.05f, 10.0f);

	ImGui::SeparatorText("Material");
	ImGui::ColorEdit3("Albedo", o.albedo);
	ImGui::ColorEdit3("Specular color", o.specular);
	ImGui::SliderFloat("Shininess", &o.shininess, 1.0f, 256.0f, "%.1f", ImGuiSliderFlags_Logarithmic);

	if (ImGui::Button("Reset object")) {
		resetObject(uint32_t(selected_object));
	}

	ImGui::End();

	ImGui::Begin("Animation");

	if (ImGui::Button(anim_playing ? "Pause" : "Play")) {
		anim_playing = !anim_playing;
	}
	ImGui::SameLine();
	if (ImGui::Button("Restart")) {
		anim_time = 0.0;
	}

	ImGui::SliderFloat("Speed", &anim_speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Trajectory radius", &anim_radius, 0.0f, 2.0f);
	ImGui::SliderFloat("Trajectory frequency", &anim_frequency, 0.1f, 5.0f);
	ImGui::SliderFloat("Spin speed (deg/s)", &anim_spin_speed, 0.0f, 360.0f);

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	const float width = float(context.swapchain_extent.width);
	const float height = float(context.swapchain_extent.height);
	const float aspect = (height > 0.0f) ? (width / height) : 1.0f;

	const float z_near = near_plane;
	const float z_far = (far_plane > z_near + 0.1f) ? far_plane : (z_near + 0.1f);

	const Mat4 projection = (projection_mode == 0)
		? perspective(radians(fov_degrees), aspect, z_near, z_far)
		: orthographic(ortho_size * aspect, ortho_size, z_near, z_far);

	const Mat4 view_projection = mul(projection, computeViewMatrix());

	float light_dir[3] = { dir_light_direction[0], dir_light_direction[1], dir_light_direction[2] };
	const float light_len = std::sqrt(light_dir[0] * light_dir[0] +
	                                  light_dir[1] * light_dir[1] +
	                                  light_dir[2] * light_dir[2]);
	if (light_len > 1e-6f) {
		for (float& c : light_dir) {
			c /= light_len;
		}
	} else {
		light_dir[0] = 0.0f; light_dir[1] = 1.0f; light_dir[2] = 0.0f;
	}

	GlobalUniforms global_uniforms{};
	std::memcpy(global_uniforms.viewProjection, view_projection.m, sizeof(global_uniforms.viewProjection));
	std::memcpy(global_uniforms.cameraPosition, camera_position, sizeof(global_uniforms.cameraPosition));
	std::memcpy(global_uniforms.directionalDirection, light_dir, sizeof(global_uniforms.directionalDirection));
	for (int c = 0; c < 3; ++c) {
		global_uniforms.directionalColor[c] = dir_light_color[c] * dir_light_intensity;
	}
	std::memcpy(global_uniforms.ambientLight, ambient_light, sizeof(global_uniforms.ambientLight));

	GpuPointLight gpu_points[max_point_lights] = {};
	const uint32_t point_count = uint32_t(point_lights.size());
	for (uint32_t i = 0; i < point_count; ++i) {
		const PointLight& src = point_lights[i];
		std::memcpy(gpu_points[i].position, src.position, sizeof(src.position));
		std::memcpy(gpu_points[i].color, src.color, sizeof(src.color));
		gpu_points[i].intensity = src.intensity;
	}

	GpuSpotLight gpu_spots[max_spot_lights] = {};
	const uint32_t spot_count = uint32_t(spot_lights.size());
	for (uint32_t i = 0; i < spot_count; ++i) {
		const SpotLight& src = spot_lights[i];
		std::memcpy(gpu_spots[i].position, src.position, sizeof(src.position));
		std::memcpy(gpu_spots[i].color, src.color, sizeof(src.color));
		normalize3(src.direction, gpu_spots[i].direction);
		gpu_spots[i].intensity = src.intensity;
		gpu_spots[i].cosInner = std::cos(radians(src.inner_angle));
		gpu_spots[i].cosOuter = std::cos(radians(src.outer_angle));
	}

	global_uniforms.pointLightCount = point_count;
	global_uniforms.spotLightCount = spot_count;
	global_uniforms.markerRadius = marker_radius;
	global_uniforms.coneLength = cone_length;
	uploadToBuffer(global_uniform_buffer, &global_uniforms, sizeof(global_uniforms));

	if (point_count > 0) {
		uploadToBuffer(point_light_buffer, gpu_points, point_count * sizeof(GpuPointLight));
	}
	if (spot_count > 0) {
		uploadToBuffer(spot_light_buffer, gpu_spots, spot_count * sizeof(GpuSpotLight));
	}

	for (int i = 0; i < object_count; ++i) {
		const Mat4 model = computeModelMatrix(uint32_t(i));

		ObjectUniforms object_uniforms{};
		std::memcpy(object_uniforms.model, model.m, sizeof(object_uniforms.model));
		std::memcpy(object_uniforms.albedo, objects[i].albedo, sizeof(object_uniforms.albedo));
		std::memcpy(object_uniforms.specular, objects[i].specular, sizeof(object_uniforms.specular));
		object_uniforms.shininess = objects[i].shininess;

		uploadToBuffer(object_uniform_buffers[i], &object_uniforms, sizeof(object_uniforms));
	}

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

	const VkClearValue clear_values[] = {
		{ .color = { .float32 = { 0.1f, 0.1f, 0.12f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .offset = { 0, 0 }, .extent = context.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = width,
		.height = height,
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor = { .offset = { 0, 0 }, .extent = context.swapchain_extent };

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.buffer, &vertex_buffer_offset);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        pipeline_layout, 0, 1, &global_descriptor_set, 0, nullptr);

	for (int i = 0; i < object_count; ++i) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        pipeline_layout, 1, 1, &object_descriptor_sets[i], 0, nullptr);
		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
	}

	if (show_light_markers && point_count + spot_count > 0) {
		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, marker_pipeline);
		vkCmdDrawIndexed(fd.command_buffer, index_count, point_count + spot_count, 0, 0, 0);
	}

	if (show_spot_cones && spot_count > 0) {
		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, cone_pipeline);
		vkCmdDraw(fd.command_buffer, cone_vertex_count, spot_count, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
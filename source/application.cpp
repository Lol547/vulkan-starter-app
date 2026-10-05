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

// r = a * b
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
	float color[3];
};

struct GlobalUniforms {
	float projection[16];
};

struct ObjectUniforms {
	float model[16];
	float tint[4];
};

constexpr uint32_t max_objects = 4;

struct SceneObject {
	float position[3];
	float rotation[3];
	float scale[3];
	float color[3];
};

#ifndef SHADER_DIR
#define SHADER_DIR "shaders/"
#endif

constexpr char vertex_shader_path[] = SHADER_DIR "shader.vert.spv";
constexpr char fragment_shader_path[] = SHADER_DIR "shader.frag.spv";

constexpr uint32_t sphere_rings = 1000;
constexpr uint32_t sphere_slices = 1000;

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};

Buffer vertex_buffer;
Buffer index_buffer;
uint32_t index_count = 0;

Buffer global_uniform_buffer;
VkDescriptorSetLayout global_set_layout = VK_NULL_HANDLE;
VkDescriptorSet global_descriptor_set = VK_NULL_HANDLE;

Buffer object_uniform_buffers[max_objects];
VkDescriptorSetLayout object_set_layout = VK_NULL_HANDLE;
VkDescriptorSet object_descriptor_sets[max_objects] = {};

VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

SceneObject objects[max_objects];
int object_count = 3;
int selected_object = 0;

int projection_mode = 0;
float fov_degrees = 60.0f;
float ortho_size = 5.0f;
float near_plane = 0.1f;
float far_plane = 100.0f;

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

	SceneObject& o = objects[i];
	std::memcpy(o.position, positions[i], sizeof(o.position));
	o.rotation[0] = o.rotation[1] = o.rotation[2] = 0.0f;
	o.scale[0] = o.scale[1] = o.scale[2] = scales[i];
	o.color[0] = o.color[1] = o.color[2] = 1.0f;
}

void generateSphere(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
	const auto makeVertex = [](float x, float y, float z) {
		return Vertex{ { x, y, z }, { x * 0.5f + 0.5f, y * 0.5f + 0.5f, z * 0.5f + 0.5f } };
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

bool createPipeline() {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = loadShaderModule(vertex_shader_path);
	VkShaderModule fragment_shader = loadShaderModule(fragment_shader_path);

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
			.offset = offsetof(Vertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = sizeof(vertex_bindings) / sizeof(vertex_bindings[0]),
		.pVertexBindingDescriptions = vertex_bindings,
		.vertexAttributeDescriptionCount = sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo assembly_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
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
		.pVertexInputState = &vertex_input_state,
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
	                                                  &pipeline_info, nullptr, &pipeline);

	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding set_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo set_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(set_bindings) / sizeof(set_bindings[0]),
		.pBindings = set_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &set_layout_info, nullptr,
	                                &global_set_layout) != VK_SUCCESS ||
	    vkCreateDescriptorSetLayout(context.device, &set_layout_info, nullptr,
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

	VkDescriptorBufferInfo buffer_infos[1 + max_objects];
	VkWriteDescriptorSet writes[1 + max_objects];

	buffer_infos[0] = {
		.buffer = global_uniform_buffer.buffer,
		.offset = 0,
		.range = sizeof(GlobalUniforms),
	};

	writes[0] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = global_descriptor_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_infos[0],
	};

	for (uint32_t i = 0; i < max_objects; ++i) {
		buffer_infos[1 + i] = {
			.buffer = object_uniform_buffers[i].buffer,
			.offset = 0,
			.range = sizeof(ObjectUniforms),
		};

		writes[1 + i] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = object_descriptor_sets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[1 + i],
		};
	}

	vkUpdateDescriptorSets(context.device, 1 + max_objects, writes, 0, nullptr);

	return true;
}

void destroyResources() {
	auto& context = graphics::internal::context;

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, global_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, object_set_layout, nullptr);

	for (uint32_t i = 0; i < max_objects; ++i) {
		destroyBuffer(object_uniform_buffers[i]);
	}
	destroyBuffer(global_uniform_buffer);
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);

	pipeline = VK_NULL_HANDLE;
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

	for (uint32_t i = 0; i < max_objects; ++i) {
		if (!createBuffer((sizeof(ObjectUniforms) + 0xf) & ~size_t(0xf),
		                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, object_uniform_buffers[i])) {
			destroyResources();
			return false;
		}
	}

	if (!createDescriptors() || !createPipeline()) {
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

	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Tint", o.color);

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

	GlobalUniforms global_uniforms{};
	std::memcpy(global_uniforms.projection, projection.m, sizeof(global_uniforms.projection));
	uploadToBuffer(global_uniform_buffer, &global_uniforms, sizeof(global_uniforms));

	for (int i = 0; i < object_count; ++i) {
		const Mat4 model = computeModelMatrix(uint32_t(i));

		ObjectUniforms object_uniforms{};
		std::memcpy(object_uniforms.model, model.m, sizeof(object_uniforms.model));
		object_uniforms.tint[0] = objects[i].color[0];
		object_uniforms.tint[1] = objects[i].color[1];
		object_uniforms.tint[2] = objects[i].color[2];
		object_uniforms.tint[3] = 1.0f;

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

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application

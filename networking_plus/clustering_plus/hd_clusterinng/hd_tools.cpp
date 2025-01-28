#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <sstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <random>
using namespace std;

// HD encoding parameters
static int HD_DIM = 2048;        
static int HD_Q = 16;            
static float HD_ID_FLIP = 2.0;
const int pack_len = (HD_DIM + 32 - 1)/32;

static float fragment_tol = 0.05;
static int max_mz = 1500;
static int min_mz = 101;

class HDVector {
private:
    std::unique_ptr<uint32_t[]> packed_bits;
    int pack_len;

public:
    HDVector() : pack_len(0) {}
    
    explicit HDVector(int len) : pack_len(len) {
        packed_bits = std::make_unique<uint32_t[]>(len);
        std::fill_n(packed_bits.get(), len, 0);
    }
    
    // Copy constructor
    HDVector(const HDVector& other) : pack_len(other.pack_len) {
        if (other.packed_bits) {
            packed_bits = std::make_unique<uint32_t[]>(pack_len);
            std::copy_n(other.packed_bits.get(), pack_len, packed_bits.get());
        }
    }
    
    // Move constructor
    HDVector(HDVector&& other) noexcept = default;
    
    // Copy assignment
    HDVector& operator=(const HDVector& other) {
        if (this != &other) {
            pack_len = other.pack_len;
            if (other.packed_bits) {
                packed_bits = std::make_unique<uint32_t[]>(pack_len);
                std::copy_n(other.packed_bits.get(), pack_len, packed_bits.get());
            } else {
                packed_bits.reset();
            }
        }
        return *this;
    }
    
    // Move assignment
    HDVector& operator=(HDVector&& other) noexcept = default;
    
    // Accessors
    const uint32_t* data() const { return packed_bits.get(); }
    uint32_t* data() { return packed_bits.get(); }
    int length() const { return pack_len; }
    bool empty() const { return !packed_bits || pack_len == 0; }
};

float hamming_distance(const uint32_t* a, const uint32_t* b, int pack_len) {
    float dist = 0;
    for(int i = 0; i < pack_len; i++) {
        dist += __builtin_popcount(a[i] ^ b[i]);
    }
    return dist/(32.0f * pack_len);
}

std::unique_ptr<float[]> generate_level_vectors(int D, int Q) {
    auto levels = std::make_unique<float[]>(D * (Q + 1));
    auto base = std::make_unique<float[]>(D);
    
    for(int i = 0; i < D; i++) {
        base[i] = (i < D/2) ? -1.0f : 1.0f;
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(base.get(), base.get() + D, g);
    
    for(int i = 0; i <= Q; i++) {
        int flip = (int)((i/(float)Q) * D) / 2;
        std::copy_n(base.get(), D, &levels[i*D]);
        for(int j = 0; j < flip; j++) {
            levels[i*D + j] *= -1;
        }
    }
    
    return levels;
}

float* generate_id_vectors(int D, int totalFeatures, float flip_factor) {
    float* id_hvs = new float[D * totalFeatures];
    int nFlip = (int)(D/flip_factor);
    
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    float* base = new float[D];
    for(int i = 0; i < D; i++) {
        base[i] = dist(gen);
    }
    
    memcpy(id_hvs, base, D * sizeof(float));
    
    for(int feat = 1; feat < totalFeatures; feat++) {
        memcpy(&id_hvs[feat*D], base, D * sizeof(float));
        
        std::vector<int> idx_to_flip(nFlip);
        for(int i = 0; i < nFlip; i++) {
            idx_to_flip[i] = rand() % D;
        }
        
        for(int idx : idx_to_flip) {
            id_hvs[feat*D + idx] *= -1;
        }
    }
    
    delete[] base;
    return id_hvs;
}

void convert_float_to_uint_vectors(float* float_vectors, uint32_t* uint_vectors, int D, int num_vectors, int pack_len) {
    memset(uint_vectors, 0, num_vectors * pack_len * sizeof(uint32_t));
    
    for(int vec = 0; vec < num_vectors; vec++) {
        for(int i = 0; i < D; i++) {
            if(float_vectors[vec * D + i] > 0) {
                uint_vectors[vec * pack_len + (i/32)] |= (1U << (31 - (i % 32)));
            }
        }
    }
}

void bit_packing(uint32_t* output, float* input, int orig_length, int pack_length, int num_vec) {
    memset(output, 0, num_vec * pack_length * sizeof(uint32_t));
    
    for(int sample_idx = 0; sample_idx < num_vec; sample_idx++) {
        for(int i = 0; i < orig_length; i++) {
            if(input[sample_idx*orig_length + i] > 0) {
                output[sample_idx*pack_length + (i/32)] |= (1U << (31 - (i % 32)));
            }
        }
    }
}

void encode_spectrum_hd(
    uint32_t* output, 
    const vector<double>& mz_values,
    const vector<double>& intensities,
    uint32_t* level_vectors,
    uint32_t* id_vectors,
    int hd_dim,
    int hd_q,
    int pack_len,
    int bin_size) {
    
    memset(output, 0, pack_len * sizeof(uint32_t));
    
    for (size_t i = 0; i < mz_values.size(); i++) {
        int level_idx = int(intensities[i] * hd_q);
        level_idx = min(level_idx, hd_q); // Ensure we don't exceed array bounds
        
        int id_idx = int(mz_values[i] / bin_size);
        
        // Bounds checking
        if (level_idx < 0 || id_idx < 0) continue;
        
        for (int j = 0; j < pack_len; j++) {
            output[j] ^= level_vectors[level_idx * pack_len + j] & 
                        id_vectors[id_idx * pack_len + j];
        }
    }
}

float get_hd_similarity(const HDVector& vec1, const HDVector& vec2) {
    if (vec1.empty() || vec2.empty()) {
        throw std::runtime_error("Null packed_bits pointer in get_hd_similarity");
    }
    if (vec1.length() != vec2.length()) {
        throw std::runtime_error("Mismatched pack_len in get_hd_similarity");
    }
    return 1.0f - hamming_distance(vec1.data(), vec2.data(), vec1.length());
}

int get_dim() {
    double bin_size = fragment_tol;
    if (bin_size <= 0) {
        throw invalid_argument("Bin size must be greater than zero.");
    }

    double start_dim = min_mz - fmod(min_mz, bin_size);
    double end_dim = max_mz + bin_size - fmod(max_mz, bin_size);
    int num_bins = static_cast<int>(ceil((end_dim - start_dim) / bin_size));

    return num_bins;
}


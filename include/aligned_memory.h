#ifndef ALIGNED_MEMORY_H
#define ALIGNED_MEMORY_H

#include <vector>
#include <cstdlib>
#include <stdexcept>

// Custom allocator for aligned memory
template<typename T>
class AlignedAllocator {
public:
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template<typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    T* allocate(size_t n) {
        void* ptr = aligned_alloc(16, n * sizeof(T)); // 16-byte alignment for NEON
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t) noexcept {
        std::free(ptr);
    }

    template<typename U>
    bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }
    template<typename U>
    bool operator!=(const AlignedAllocator<U>&) const noexcept { return false; }
};

// Alias for aligned vector
template<typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;

#endif // ALIGNED_MEMORY_H

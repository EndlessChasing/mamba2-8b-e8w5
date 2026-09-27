// Chunked canonical-Huffman W4 packer and full integer-code verifier.
// Called from pack_w4_entropy_base.py through ctypes.
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" int pack_verify_w4(
    const int8_t* symbols, size_t count, const uint16_t* codes,
    const uint8_t* lengths, uint8_t* output, size_t capacity,
    uint32_t* offsets, size_t chunk_values, size_t* output_size) {
  if (!symbols || !codes || !lengths || !output || !offsets || !output_size ||
      chunk_values == 0 || count % chunk_values != 0) return 1;
  size_t used = 0;
  const size_t chunks = count / chunk_values;
  for (size_t chunk = 0; chunk < chunks; ++chunk) {
    if (used > UINT32_MAX) return 2;
    offsets[chunk] = static_cast<uint32_t>(used);
    uint32_t accumulator = 0;
    unsigned width = 0;
    const size_t first = chunk * chunk_values;
    for (size_t position = first; position < first + chunk_values; ++position) {
      const int index = static_cast<int>(symbols[position]) + 7;
      if (index < 0 || index >= 15 || lengths[index] == 0 || lengths[index] > 15)
        return 3;
      accumulator = (accumulator << lengths[index]) | codes[index];
      width += lengths[index];
      while (width >= 8) {
        width -= 8;
        if (used >= capacity) return 4;
        output[used++] = static_cast<uint8_t>(accumulator >> width);
        accumulator &= (1u << width) - 1u;
      }
    }
    if (width != 0) {
      if (used >= capacity) return 4;
      output[used++] = static_cast<uint8_t>(accumulator << (8 - width));
    }
  }
  *output_size = used;

  // Decode the entire emitted bitstream and compare every integer to its
  // source. Padding bits after the final codeword in each chunk are ignored.
  int8_t decode[16][1 << 15] = {};
  for (unsigned length = 0; length < 16; ++length)
    for (unsigned code = 0; code < (1u << 15); ++code) decode[length][code] = 127;
  for (int index = 0; index < 15; ++index)
    decode[lengths[index]][codes[index]] = static_cast<int8_t>(index - 7);
  for (size_t chunk = 0; chunk < chunks; ++chunk) {
    const size_t start = offsets[chunk];
    const size_t end = chunk + 1 < chunks ? offsets[chunk + 1] : used;
    size_t recovered = 0;
    unsigned current = 0;
    unsigned width = 0;
    for (size_t byte = start; byte < end && recovered < chunk_values; ++byte) {
      for (int bit = 7; bit >= 0 && recovered < chunk_values; --bit) {
        current = (current << 1) | ((output[byte] >> bit) & 1);
        ++width;
        if (width >= 16) return 5;
        const int8_t candidate = decode[width][current];
        if (candidate != 127) {
          if (candidate != symbols[chunk * chunk_values + recovered]) return 6;
          ++recovered;
          current = width = 0;
        }
      }
    }
    if (recovered != chunk_values) return 7;
  }
  return 0;
}

extern "C" int pack_verify_scale_u16(
    const uint16_t* symbols, size_t count, const uint32_t* codes,
    const uint8_t* lengths, uint8_t* output, size_t capacity,
    uint32_t* offsets, size_t chunk_values, size_t* output_size) {
  if (!symbols || !codes || !lengths || !output || !offsets || !output_size ||
      chunk_values == 0 || count % chunk_values != 0) return 1;
  size_t used = 0;
  const size_t chunks = count / chunk_values;
  for (size_t chunk = 0; chunk < chunks; ++chunk) {
    if (used > UINT32_MAX) return 2;
    offsets[chunk] = static_cast<uint32_t>(used);
    uint64_t accumulator = 0;
    unsigned width = 0;
    const size_t first = chunk * chunk_values;
    for (size_t position = first; position < first + chunk_values; ++position) {
      const uint16_t symbol = symbols[position];
      const unsigned length = lengths[symbol];
      if (length == 0 || length > 32) return 3;
      accumulator = (accumulator << length) | codes[symbol];
      width += length;
      while (width >= 8) {
        width -= 8;
        if (used >= capacity) return 4;
        output[used++] = static_cast<uint8_t>(accumulator >> width);
        accumulator &= (uint64_t(1) << width) - 1;
      }
    }
    if (width != 0) {
      if (used >= capacity) return 4;
      output[used++] = static_cast<uint8_t>(accumulator << (8 - width));
    }
  }
  *output_size = used;

  struct Node {
    int child[2] = {-1, -1};
    int symbol = -1;
  };
  std::vector<Node> tree(1);
  tree.reserve(65536);
  for (unsigned symbol = 0; symbol < 65536; ++symbol) {
    const unsigned length = lengths[symbol];
    if (length == 0) continue;
    int node = 0;
    for (int shift = static_cast<int>(length) - 1; shift >= 0; --shift) {
      const unsigned bit = (codes[symbol] >> shift) & 1u;
      int next = tree[node].child[bit];
      if (next < 0) {
        next = static_cast<int>(tree.size());
        tree[node].child[bit] = next;
        tree.emplace_back();
      }
      node = next;
    }
    if (tree[node].symbol >= 0) return 5;
    tree[node].symbol = static_cast<int>(symbol);
  }
  for (size_t chunk = 0; chunk < chunks; ++chunk) {
    const size_t start = offsets[chunk];
    const size_t end = chunk + 1 < chunks ? offsets[chunk + 1] : used;
    size_t recovered = 0;
    int node = 0;
    for (size_t byte = start; byte < end && recovered < chunk_values; ++byte) {
      for (int bit = 7; bit >= 0 && recovered < chunk_values; --bit) {
        const unsigned side = (output[byte] >> bit) & 1u;
        node = tree[node].child[side];
        if (node < 0) return 6;
        const int candidate = tree[node].symbol;
        if (candidate >= 0) {
          if (candidate != symbols[chunk * chunk_values + recovered]) return 7;
          ++recovered;
          node = 0;
        }
      }
    }
    if (recovered != chunk_values) return 8;
  }
  return 0;
}

// Standalone decoder used to verify serialized E8/W5 Huffman containers.
extern "C" int unpack_huffman_u16(
    const uint8_t* input, size_t input_size, const uint32_t* offsets,
    size_t count, size_t chunk_values, const uint32_t* codes,
    const uint8_t* lengths, uint16_t* output) {
  if (!input || !offsets || !codes || !lengths || !output ||
      !chunk_values || count % chunk_values) return 1;
  struct Node { int child[2] = {-1, -1}; int symbol = -1; };
  std::vector<Node> tree(1);
  for (unsigned symbol = 0; symbol < 65536; ++symbol) {
    const unsigned length = lengths[symbol];
    if (!length) continue;
    if (length > 32) return 2;
    int node = 0;
    for (int shift = static_cast<int>(length)-1; shift >= 0; --shift) {
      unsigned side = (codes[symbol] >> shift) & 1u;
      int next = tree[node].child[side];
      if (next < 0) {
        next = static_cast<int>(tree.size());
        tree[node].child[side] = next;
        tree.emplace_back();
      }
      node = next;
    }
    if (tree[node].symbol >= 0) return 3;
    tree[node].symbol = symbol;
  }
  const size_t chunks = count / chunk_values;
  for (size_t chunk = 0; chunk < chunks; ++chunk) {
    const size_t start = offsets[chunk];
    const size_t end = chunk+1 < chunks ? offsets[chunk+1] : input_size;
    if (start > end || end > input_size) return 4;
    size_t recovered = 0;
    int node = 0;
    for (size_t byte = start; byte < end && recovered < chunk_values; ++byte) {
      for (int bit = 7; bit >= 0 && recovered < chunk_values; --bit) {
        node = tree[node].child[(input[byte] >> bit) & 1u];
        if (node < 0) return 5;
        if (tree[node].symbol >= 0) {
          output[chunk*chunk_values+recovered++] = tree[node].symbol;
          node = 0;
        }
      }
    }
    if (recovered != chunk_values) return 6;
  }
  return 0;
}

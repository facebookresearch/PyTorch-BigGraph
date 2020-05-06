// Copyright 2004-present Facebook. All Rights Reserved.

#include <atomic>

#include <torch/extension.h>

namespace py = pybind11;

namespace at {
  class CPUGenerator;
  class CPUGeneratorImpl;
};

torch::Tensor randperm(long numItems, int numThreads, int64_t seedIn = -1) {
  // workaround a breaking chang in the name of CPUGenerator in PyTorch 1.5
  // https://github.com/pytorch/pytorch/pull/36027
  // This code will pick whichever class exists
  typedef std::conditional<std::is_constructible<at::CPUGeneratorImpl, uint64_t>::value, at::CPUGeneratorImpl, at::CPUGenerator>::type CPUGeneratorType;

  auto perm = torch::empty(numItems, torch::kInt64);
  auto permAccessor = perm.accessor<int64_t, 1>();
  assert(numThreads < 256);
  torch::Tensor chunks = torch::empty({numItems}, torch::kUInt8);
  auto chunksAccessor = chunks.accessor<uint8_t, 1>();
  std::vector<std::vector<int>> allCounts(numThreads);
  auto stepOne = [&](int64_t startIdx, int64_t endIdx, int threadIdx) {
    CPUGeneratorType generator(seedIn >= 0 ? seedIn + threadIdx : at::default_rng_seed_val);

    std::vector<int>& myCounts = allCounts[threadIdx];
    myCounts.assign(numThreads, 0);
    for (int idx = startIdx; idx < endIdx; idx += 1) {
      chunksAccessor[idx] = generator.random() % numThreads;
      myCounts[chunksAccessor[idx]] += 1;
    }
  };
  std::vector<std::thread> stepOneThreads;
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepOneThreads.emplace_back(
        stepOne,
        threadIdx * numItems / numThreads,
        (threadIdx + 1) * numItems / numThreads,
        threadIdx);
  }
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepOneThreads[threadIdx].join();
  }
  std::vector<std::vector<int>> allOffsets(numThreads);
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    allOffsets[threadIdx].reserve(numThreads);
  }
  int64_t offset = 0;
  for (int chunkIdx = 0; chunkIdx < numThreads; chunkIdx += 1) {
    for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
      allOffsets[threadIdx].push_back(offset);
      offset += allCounts[threadIdx][chunkIdx];
    }
  }
  assert(offset == numItems);
  auto stepTwo = [&](int64_t startIdx, int64_t endIdx, int threadIdx) {
    std::vector<int>& myOffsets = allOffsets[threadIdx];
    for (int idx = startIdx; idx < endIdx; idx += 1) {
      int& offset = myOffsets[chunksAccessor[idx]];
      permAccessor[offset] = idx;
      offset += 1;
    }
  };
  std::vector<std::thread> stepTwoThreads;
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepTwoThreads.emplace_back(
        stepTwo,
        threadIdx * numItems / numThreads,
        (threadIdx + 1) * numItems / numThreads,
        threadIdx);
  }
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepTwoThreads[threadIdx].join();
  }
  auto stepThree = [&](int64_t startIdx, int64_t endIdx, int threadIdx) {
    CPUGeneratorType generator(seedIn >= 0 ? seedIn + threadIdx + numThreads: at::default_rng_seed_val);
    for (int idx = startIdx; idx < endIdx - 1; idx += 1) {
      int64_t otherIdx = idx + generator.random() % (endIdx - idx);
      std::swap(permAccessor[idx], permAccessor[otherIdx]);
    }
  };
  std::vector<std::thread> stepThreeThreads;
  offset = 0;
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepThreeThreads.emplace_back(
        stepThree, offset, allOffsets[numThreads - 1][threadIdx], threadIdx);
    offset = allOffsets[numThreads - 1][threadIdx];
  }
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepThreeThreads[threadIdx].join();
  }
  return perm;
}

torch::Tensor reversePermutation(const torch::Tensor& perm, int numThreads) {
  auto permAccessor = perm.accessor<int64_t, 1>();
  auto numItems = permAccessor.size(0);
  torch::Tensor res = torch::empty({numItems}, torch::kInt64);
  auto resAccessor = res.accessor<int64_t, 1>();
  auto thread = [&](int64_t startIdx, int64_t endIdx) {
    for (int64_t idx = startIdx; idx < endIdx; idx += 1) {
      resAccessor[permAccessor[idx]] = idx;
    }
  };
  std::vector<std::thread> threads;
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    threads.emplace_back(
        thread,
        threadIdx * numItems / numThreads,
        (threadIdx + 1) * numItems / numThreads);
  }
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    threads[threadIdx].join();
  }
  return res;
}

void shuffle(
    at::Tensor& tensor,
    const at::Tensor& permutation,
    int numThreads) {
  if (permutation.scalar_type() != c10::ScalarType::Long) {
    throw std::invalid_argument("Permutation must have int64 dtype");
  }
  if (permutation.dim() != 1) {
    throw std::invalid_argument("Permutation must have exactly one dimension");
  }
  if (tensor.dim() < 1) {
    throw std::invalid_argument("Tensor must have at least one dimension");
  }
  int64_t numRows = tensor.sizes()[0];
  if (numRows != permutation.sizes()[0]) {
    throw std::invalid_argument(
        "Tensor and permutation must have the same number of elements on the first dimension");
  }
  if (numRows == 0) {
    return;
  }
  int64_t rowStride = tensor.strides()[0] * tensor.element_size();
  if (rowStride == 0) {
    return;
  }
  if (!tensor[0].is_contiguous()) {
    throw std::invalid_argument(
        "Each sub-tensor of tensor (along the first dimension) must be contiguous");
  }
  for (int i = 1; i < tensor.dim(); i += 1) {
    if (tensor.strides()[i] == 0) {
      throw std::invalid_argument(
          "Tensor cannot have strides that are zero (for now)");
    }
  }
  int64_t rowSize = tensor[0].nbytes();

  // This pointer's type doesn't matter, as long as it has size 1.
  uint8_t* tensorData = reinterpret_cast<uint8_t*>(tensor.data_ptr());
  int64_t* permutationData = permutation.data_ptr<int64_t>();

  std::vector<std::atomic_flag> checks(numRows);
  std::atomic_flag* checksData = checks.data();

  auto stepOne = [&](int64_t startIdx, int64_t endIdx) {
    for (int64_t idx = startIdx; idx < endIdx; idx += 1) {
      checksData[idx].clear();
    }
  };
  std::vector<std::thread> stepOneThreads;
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepOneThreads.emplace_back(
        stepOne,
        threadIdx * numRows / numThreads,
        (threadIdx + 1) * numRows / numThreads);
  }
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepOneThreads[threadIdx].join();
  }
  auto stepTwo = [&](int64_t startIdx, int64_t endIdx) {
    std::vector<uint8_t> bufferOne(rowSize);
    std::vector<uint8_t> bufferTwo(rowSize);
    void* bufferOneData = bufferOne.data();
    void* bufferTwoData = bufferTwo.data();
    for (int64_t baseIdx = startIdx; baseIdx < endIdx; baseIdx += 1) {
      int64_t curIdx = baseIdx;
      std::memcpy(bufferOneData, tensorData + curIdx * rowStride, rowSize);
      if (checksData[curIdx].test_and_set()) {
        continue;
      }
      bool done = false;
      while (!done) {
        curIdx = permutationData[curIdx];
        if (curIdx < 0 || curIdx >= numRows) {
          throw std::invalid_argument("Permutation has out-of-bound values");
        }
        std::memcpy(bufferTwoData, tensorData + curIdx * rowStride, rowSize);
        done = checksData[curIdx].test_and_set();
        std::memcpy(tensorData + curIdx * rowStride, bufferOneData, rowSize);
        std::swap(bufferOneData, bufferTwoData);
      }
    }
  };
  std::vector<std::thread> stepTwoThreads;
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepTwoThreads.emplace_back(
        stepTwo,
        threadIdx * numRows / numThreads,
        (threadIdx + 1) * numRows / numThreads);
  }
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepTwoThreads[threadIdx].join();
  }
}


/**
 * This function takes an edgelist (representing a bucket) and splits it into
 * N*M sub-edgelists (its subbuckets). The input edgelist is given as three
 * 1-dimensional vectors of the same length, `lhsIn`, `rhsIn` and `relIn`, using
 * the same format as in Python (the first two contain the offset of the left-
 * and right-hand side entities, the third the type of the relation). The
 * values of N and M above are given by `numLhsSubParts` and `numRhsSubParts`.
 * The return type is a dict that has as keys all the pairs (i, j) for
 * 0 <= i < N and 0 <= j < M, and as values the triples lhs, rhs and rel
 * representing the subbucket.
 *
 * Entities are assigned to subpartitions uniformly at random, and this is
 * implemented as follows: a (random) permutation is provided for each entity
 * type and this function will map an entity's offset through that permutation
 * and if the output falls into the first 1/N (or 1/M) the entity is assigned
 * to subpartition 0, if it's in the second 1/N to partition 1, and so on.
 * Those partitions are provided using `lhsPerms` and `rhsPerms`. The i-th
 * element of `lhsPerms` is the permutation to be used for the left-hand side
 * entities of relation type i. Since the same entity type may appear on
 * multiple relation types, on different sides, it's expected that the same
 * permutation will appear multiple times in `lhsPerms` and/or `rhsPerms`.
 * (Since `at::Tensor`s are just references to the underlying data, this will
 * not cause copies). In a similar way, `lhsEntityCounts[i]` and
 * `rhsEntityCounts[i]` contain the count of the entity type that appears on
 * the left- and right-hand side of relation type i. (TODO: remove these
 * arguments as they could be inferred from the permutations).
 *
 * This function does _not_ operate in-place. Its return value is a dictionary
 * containing tensors, but all these tensors will just be views over a single
 * underlying storage. Such a storage must be provided by the user, through the
 * `lhsOut`, `rhsOut` and `relOut` parameters, which must have the same shapes
 * as their `In` counterparts. This is done for efficiency, to avoid multiple
 * allocations and to permit re-use of these allocations if they are expensive
 * to make (shared memory, pinned memory, ...).
 *
 * Lastly, this function operates in parallel, and the number of parallel
 * threads can be controlled with the `numThreads` argument.
 */
std::map<
    std::pair<int8_t, int8_t>,
    std::tuple<at::Tensor, at::Tensor, at::Tensor>>
subBucket(
    const at::Tensor& lhsIn,
    const at::Tensor& rhsIn,
    const at::Tensor& relIn,
    const std::vector<int64_t>& lhsEntityCounts,
    const std::vector<at::Tensor>& lhsPerms,
    const std::vector<int64_t>& rhsEntityCounts,
    const std::vector<at::Tensor>& rhsPerms,
    at::Tensor& lhsOut,
    at::Tensor& rhsOut,
    at::Tensor& relOut,
    int8_t numLhsSubParts,
    int8_t numRhsSubParts,
    int numThreads,
    bool dynamicRelations) {
  int64_t numEdges = relIn.sizes()[0];
  size_t numRelations = lhsPerms.size();

  if (
      rhsPerms.size() != numRelations ||
      lhsEntityCounts.size() != numRelations ||
      rhsEntityCounts.size() != numRelations
  ) {
    throw std::runtime_error("Inconsistent num_relations");
  }

  std::vector<int64_t*> lhsPermsData;
  lhsPermsData.reserve(lhsPerms.size());
  for (const auto& p : lhsPerms) {
    lhsPermsData.push_back(p.data_ptr<int64_t>());
  }
  std::vector<int64_t*> rhsPermsData;
  rhsPermsData.reserve(rhsPerms.size());
  for (const auto& p : rhsPerms) {
    rhsPermsData.push_back(p.data_ptr<int64_t>());
  }
  std::vector<int64_t> lhsSubPartSizes;
  lhsSubPartSizes.reserve(lhsEntityCounts.size());
  for (const auto& c : lhsEntityCounts) {
    int64_t subPartSize = c / numLhsSubParts;
    if (c % numLhsSubParts != 0) {
      subPartSize += 1;
    }
    lhsSubPartSizes.push_back(subPartSize);
  }
  std::vector<int64_t> rhsSubPartSizes;
  rhsSubPartSizes.reserve(rhsEntityCounts.size());
  for (const auto& c : rhsEntityCounts) {
    int64_t subPartSize = c / numRhsSubParts;
    if (c % numRhsSubParts != 0) {
      subPartSize += 1;
    }
    rhsSubPartSizes.push_back(subPartSize);
  }

  // Get pointers to all the tensors for faster access.
  // TODO check type and sizes first;
  int64_t* lhsInData = lhsIn.data_ptr<int64_t>();
  int64_t* rhsInData = rhsIn.data_ptr<int64_t>();
  int64_t* relInData = relIn.data_ptr<int64_t>();
  int64_t* lhsOutData = lhsOut.data_ptr<int64_t>();
  int64_t* rhsOutData = rhsOut.data_ptr<int64_t>();
  int64_t* relOutData = relOut.data_ptr<int64_t>();
  int64_t** lhsPermsDataData = lhsPermsData.data();
  int64_t** rhsPermsDataData = rhsPermsData.data();
  int64_t* lhsSubPartSizesData = lhsSubPartSizes.data();
  int64_t* rhsSubPartSizesData = rhsSubPartSizes.data();

  std::vector<std::vector<int64_t>> subBucketCountsOrOffsets(
      numThreads, std::vector<int64_t>(numLhsSubParts * numRhsSubParts));
  std::vector<int64_t*> subBucketCountsOrOffsetsData;
  subBucketCountsOrOffsetsData.reserve(subBucketCountsOrOffsets.size());
  for (const auto& v : subBucketCountsOrOffsets) {
    subBucketCountsOrOffsetsData.push_back(const_cast<int64_t*>(v.data()));
  }
  int64_t** subBucketCountsOrOffsetsDataData =
      subBucketCountsOrOffsetsData.data();
  auto stepOne = [&](int64_t startEdgeIdx,
                     int64_t endEdgeIdx,
                     int64_t* myNumEdgesBySubBucketData) {
    for (int64_t edgeIdx = startEdgeIdx; edgeIdx < endEdgeIdx; edgeIdx += 1) {
      int64_t relId = dynamicRelations ? 0 : relInData[edgeIdx];
      if (relId >= numRelations) {
        throw std::runtime_error("rel > numRelations");
      }
      int64_t lhsOffset = lhsPermsDataData[relId][lhsInData[edgeIdx]];
      int64_t rhsOffset = rhsPermsDataData[relId][rhsInData[edgeIdx]];
      int8_t lhsSubPart = lhsOffset / lhsSubPartSizesData[relId];
      int8_t rhsSubPart = rhsOffset / rhsSubPartSizesData[relId];
      myNumEdgesBySubBucketData[lhsSubPart * numRhsSubParts + rhsSubPart] += 1;
    }
  };
  std::vector<std::thread> stepOneThreads;
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepOneThreads.emplace_back(
        stepOne,
        threadIdx * numEdges / numThreads,
        (threadIdx + 1) * numEdges / numThreads,
        subBucketCountsOrOffsetsDataData[threadIdx]);
  }
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepOneThreads[threadIdx].join();
  }

  std::map<
      std::pair<int8_t, int8_t>,
      std::tuple<at::Tensor, at::Tensor, at::Tensor>>
      res;
  int64_t offset = 0;
  for (int8_t lhsSubPart = 0; lhsSubPart < numLhsSubParts; lhsSubPart += 1) {
    for (int8_t rhsSubPart = 0; rhsSubPart < numRhsSubParts; rhsSubPart += 1) {
      int16_t subBucketIdx = lhsSubPart * numRhsSubParts + rhsSubPart;
      int64_t start = offset;
      int64_t stop = start;
      for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
        int64_t thisNumEdges =
            subBucketCountsOrOffsetsDataData[threadIdx][subBucketIdx];
        subBucketCountsOrOffsetsDataData[threadIdx][subBucketIdx] = offset;
        offset += thisNumEdges;
        stop += thisNumEdges;
      }
      at::Tensor lhs = lhsOut.slice(0, start, stop, 1);
      at::Tensor rhs = rhsOut.slice(0, start, stop, 1);
      at::Tensor rel = relOut.slice(0, start, stop, 1);
      res[std::make_pair(lhsSubPart, rhsSubPart)] =
          std::make_tuple(lhs, rhs, rel);
    }
  }
  assert(offset == numEdges);

  auto stepTwo = [&](int64_t startEdgeIdx,
                     int64_t endEdgeIdx,
                     int64_t* mySubBucketOffsets) {
    for (int64_t edgeIdx = startEdgeIdx; edgeIdx < endEdgeIdx; edgeIdx += 1) {
      int64_t relId = dynamicRelations ? 0 : relInData[edgeIdx];
      int64_t lhsOffset = lhsPermsDataData[relId][lhsInData[edgeIdx]];
      int64_t rhsOffset = rhsPermsDataData[relId][rhsInData[edgeIdx]];
      int8_t lhsSubPart = lhsOffset / lhsSubPartSizesData[relId];
      int8_t rhsSubPart = rhsOffset / rhsSubPartSizesData[relId];
      int64_t lhsSubOffset = lhsOffset % lhsSubPartSizesData[relId];
      int64_t rhsSubOffset = rhsOffset % rhsSubPartSizesData[relId];
      int64_t& offset =
          mySubBucketOffsets[lhsSubPart * numRhsSubParts + rhsSubPart];
      lhsOutData[offset] = lhsSubOffset;
      rhsOutData[offset] = rhsSubOffset;
      relOutData[offset] = relInData[edgeIdx];
      offset += 1;
    }
  };
  std::vector<std::thread> stepTwoThreads;
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepTwoThreads.emplace_back(
        stepTwo,
        threadIdx * numEdges / numThreads,
        (threadIdx + 1) * numEdges / numThreads,
        subBucketCountsOrOffsetsDataData[threadIdx]);
  }
  for (int threadIdx = 0; threadIdx < numThreads; threadIdx += 1) {
    stepTwoThreads[threadIdx].join();
  }

  return res;
}

PYBIND11_MODULE(_C, m) {
  m.def("randperm", &randperm, py::arg("num_items"), py::arg("num_threads"), py::arg("seed")=-1);
  m.def(
      "reverse_permutation",
      &reversePermutation,
      py::arg("perm"),
      py::arg("num_threads"));
  m.def(
      "shuffle",
      &shuffle,
      py::arg("tensor"),
      py::arg("permutation"),
      py::arg("num_threads"));
  m.def(
      "sub_bucket",
      &subBucket,
      py::arg("lhs_in"),
      py::arg("rhs_in"),
      py::arg("rel_in"),
      py::arg("lhs_entity_counts"),
      py::arg("lhs_perms"),
      py::arg("rhs_entity_count"),
      py::arg("rhs_perms"),
      py::arg("lhs_out"),
      py::arg("rhs_out"),
      py::arg("rel_out"),
      py::arg("num_lhs_sub_parts"),
      py::arg("num_rhs_sub_parts"),
      py::arg("num_threads"),
      py::arg("dynamic_relations"));
}

#ifndef COBWEB_DISCRETE_TYPES_H
#define COBWEB_DISCRETE_TYPES_H

#include <unordered_map>
#include <unordered_set>
#include <utility>

enum class Operation : int { BEST=0, NEW=1, MERGE=2, SPLIT=3 };

using ATTR_TYPE  = int;
using VALUE_TYPE = int;
using COUNT_TYPE = double;

using VALUE_MAP  = std::unordered_map<VALUE_TYPE, COUNT_TYPE>;
using INSTANCE_TYPE =
    std::unordered_map<ATTR_TYPE, VALUE_MAP>;

using VAL_COUNT_TYPE  = VALUE_MAP;
using AV_COUNT_TYPE   = std::unordered_map<ATTR_TYPE, VALUE_MAP>;
using AV_KEY_TYPE     = std::unordered_map<ATTR_TYPE, std::unordered_set<VALUE_TYPE>>;
using ATTR_COUNT_TYPE = std::unordered_map<ATTR_TYPE, COUNT_TYPE>;
using OPERATION_TYPE  = std::pair<double, Operation>;
using REF_ATTR_SET    = std::unordered_set<ATTR_TYPE>;

#endif // COBWEB_DISCRETE_TYPES_H

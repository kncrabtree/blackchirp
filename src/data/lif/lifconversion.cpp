#include <data/lif/lifconversion.h>

#include <QMetaEnum>

using namespace Qt::Literals::StringLiterals;
using namespace BC::LifConv;

namespace {

using Coeffs = BC::LifConv::detail::AffineCoeffs;

/// Visitation marks for the whole-graph cycle check.
enum class VisitState { Unvisited, Visiting, Done };

/// The enum key name, drawn from the Q_ENUM_NS meta-object so error
/// messages share the single source of truth with the by-name persistence
/// (BC::CSV::enumFromVariant) that these enums exist for.
QString opName(Op op)
{
    return QString::fromLatin1(QMetaEnum::fromType<Op>().valueToKey(static_cast<int>(op)));
}

/// Depth-first cycle check over the whole node map (not just nodes
/// reachable from FINAL), so a malformed-but-unreachable branch is still
/// reported. Stage-typed inputs are the graph's edges.
bool checkAcyclic(const QString &key, const std::map<QString,Node> &nodes,
                   std::map<QString,VisitState> &state, QString &err)
{
    auto &mark = state[key];
    if(mark == VisitState::Done)
        return true;
    if(mark == VisitState::Visiting)
    {
        err = u"Conversion graph contains a cycle at stage \"%1\"."_s.arg(key);
        return false;
    }

    mark = VisitState::Visiting;
    for(const auto &input : nodes.at(key).inputs)
    {
        if(input.type == RefType::Stage && !checkAcyclic(input.stageKey,nodes,state,err))
            return false;
    }
    state[key] = VisitState::Done;
    return true;
}

}

namespace {

bool resolveRef(const InputRef &ref, const std::map<QString,Node> &nodes,
                 std::map<QString,Coeffs> &cache, Coeffs &out, QString &err);

bool resolveNode(const QString &key, const std::map<QString,Node> &nodes,
                  std::map<QString,Coeffs> &cache, Coeffs &out, QString &err)
{
    auto cIt = cache.find(key);
    if(cIt != cache.end())
    {
        out = cIt->second;
        return true;
    }

    const Node &node = nodes.at(key);
    Coeffs result;
    if(node.op == Op::NHG)
    {
        Coeffs c0;
        if(!resolveRef(node.inputs[0],nodes,cache,c0,err))
            return false;
        result = Coeffs{node.n*c0.a, node.n*c0.b};
    }
    else
    {
        Coeffs c0, c1;
        if(!resolveRef(node.inputs[0],nodes,cache,c0,err))
            return false;
        if(!resolveRef(node.inputs[1],nodes,cache,c1,err))
            return false;
        if(node.op == Op::SFG)
            result = Coeffs{c0.a+c1.a, c0.b+c1.b};
        else
            result = Coeffs{c0.a-c1.a, c0.b-c1.b};
    }

    cache.emplace(key,result);
    out = result;
    return true;
}

bool resolveRef(const InputRef &ref, const std::map<QString,Node> &nodes,
                 std::map<QString,Coeffs> &cache, Coeffs &out, QString &err)
{
    switch(ref.type)
    {
    case RefType::Laser:
        out = Coeffs{1.0,0.0};
        return true;
    case RefType::Fixed:
        out = Coeffs{0.0,ref.fixedCm1};
        return true;
    case RefType::Stage:
        return resolveNode(ref.stageKey,nodes,cache,out,err);
    }
    err = u"Unrecognized input reference type."_s;
    return false;
}

}

LifConversion::LifConversion()
{
}

LifConversion::AssemblyResult LifConversion::assemble(const std::vector<BC::LifConv::Node> &nodes)
{
    AssemblyResult result;

    if(nodes.empty())
    {
        result.ok = true;
        result.conversion = LifConversion();
        return result;
    }

    std::map<QString,Node> nodeMap;
    for(const auto &node : nodes)
    {
        if(nodeMap.contains(node.stageKey))
        {
            result.errorString = u"Duplicate conversion node for stage \"%1\"."_s.arg(node.stageKey);
            return result;
        }
        nodeMap.emplace(node.stageKey,node);
    }

    QString finalKey;
    int finalCount = 0;
    for(const auto &[key,node] : nodeMap)
    {
        std::size_t expected = (node.op == Op::NHG) ? 1 : 2;
        if(node.inputs.size() != expected)
        {
            result.errorString = u"Stage \"%1\": %2 requires exactly %3 input(s), got %4."_s
                    .arg(key,opName(node.op)).arg(int(expected)).arg(int(node.inputs.size()));
            return result;
        }
        if(node.op == Op::NHG && node.n < 1)
        {
            result.errorString = u"Stage \"%1\": NHG harmonic order must be >= 1."_s.arg(key);
            return result;
        }
        for(const auto &input : node.inputs)
        {
            if(input.type == RefType::Stage && !nodeMap.contains(input.stageKey))
            {
                result.errorString = u"Stage \"%1\" references unknown stage \"%2\"."_s.arg(key,input.stageKey);
                return result;
            }
        }
        if(node.isFinal)
        {
            ++finalCount;
            finalKey = key;
        }
    }

    if(finalCount == 0)
    {
        result.errorString = u"Conversion graph has no stage marked as the FINAL (output) beam."_s;
        return result;
    }
    if(finalCount > 1)
    {
        result.errorString = u"Conversion graph has more than one stage marked as the FINAL (output) beam."_s;
        return result;
    }

    // Cycle check over every node, not just those reachable from FINAL, so
    // a malformed but currently-unreachable branch is still reported.
    std::map<QString,VisitState> visitState;
    for(const auto &[key,node] : nodeMap)
    {
        QString cycleErr;
        if(!checkAcyclic(key,nodeMap,visitState,cycleErr))
        {
            result.errorString = cycleErr;
            return result;
        }
    }

    // The graph is now known well-formed and acyclic; resolve every node's
    // output as an affine expression of the tunable fundamental,
    // value = a*fundamentalCm1 + b, via memoized recursion.
    std::map<QString,Coeffs> cache;
    QString err;
    Coeffs outputCoeffs;
    if(!resolveNode(finalKey,nodeMap,cache,outputCoeffs,err))
    {
        // Unreachable given the validation above; kept defensive.
        result.errorString = err;
        return result;
    }

    // Scope boundary (plan §1): the day-1 solver supports exactly one
    // tunable scan source. The Node/InputRef schema has no way to name a
    // second, independent tunable source in the first place (RefType::Laser
    // always denotes the single active LifLaser's fundamental, wherever it
    // is referenced in the graph), so the only violation representable
    // today is the FINAL beam ending up with zero net dependence on that
    // fundamental — e.g. a DFG of two equal multiples of it, or a graph
    // built entirely from Fixed sources. Reject that here as "not exactly
    // one tunable source"; genuine support for a second, independently
    // tunable source requires both a richer InputRef/RefType and a
    // multi-axis LifConfig acquisition model, deferred as a unit.
    if(outputCoeffs.a == 0.0)
    {
        result.errorString = u"Conversion graph FINAL beam has no net dependence on the "
                              "tunable laser source (exactly one tunable source is required)."_s;
        return result;
    }

    LifConversion conv;
    conv.d_identity = false;
    conv.d_output = outputCoeffs;
    for(const auto &[key,node] : nodeMap)
    {
        Coeffs primary;
        if(!resolveRef(node.inputs[0],nodeMap,cache,primary,err))
        {
            // Unreachable given the validation above; kept defensive.
            result.errorString = err;
            return result;
        }
        conv.d_primaryInput.emplace(key,primary);
    }

    result.ok = true;
    result.conversion = conv;
    return result;
}

double LifConversion::laserToOutput(double fundamentalCm1) const
{
    return d_output.a*fundamentalCm1 + d_output.b;
}

double LifConversion::outputToLaser(double outputCm1) const
{
    return (outputCm1 - d_output.b)/d_output.a;
}

double LifConversion::stageInput(const QString &stageKey, double fundamentalCm1) const
{
    auto it = d_primaryInput.find(stageKey);
    if(it == d_primaryInput.end())
        return -1.0;
    return it->second.a*fundamentalCm1 + it->second.b;
}

std::pair<double,double> LifConversion::outputRange(double laserMinCm1, double laserMaxCm1) const
{
    double lo = laserToOutput(laserMinCm1);
    double hi = laserToOutput(laserMaxCm1);
    if(lo <= hi)
        return {lo,hi};
    return {hi,lo};
}

bool LifConversion::isIdentity() const
{
    return d_identity;
}

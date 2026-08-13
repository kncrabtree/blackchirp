#include <data/loadout/lifconversionsnapshot.h>

LifConversionSnapshot LifConversionSnapshot::fromNodes(const std::vector<BC::LifConv::Node> &nodes,
                                                        const QString &laserKey)
{
    LifConversionSnapshot snap;
    snap.laserKey = laserKey;
    snap.wiring.reserve(nodes.size());
    for(const auto &node : nodes)
    {
        BC::LifConv::StageWiring w;
        w.stageKey = node.stageKey;
        w.inputs = node.inputs;
        w.isFinal = node.isFinal;
        snap.wiring.push_back(std::move(w));
    }
    return snap;
}

std::vector<BC::LifConv::Node> LifConversionSnapshot::toNodes(
    const std::function<BC::LifConv::Op(const QString&)> &opOf,
    const std::function<int(const QString&)> &harmonicOf) const
{
    std::vector<BC::LifConv::Node> nodes;
    nodes.reserve(wiring.size());
    for(const auto &w : wiring)
    {
        BC::LifConv::Node node;
        node.stageKey = w.stageKey;
        node.inputs = w.inputs;
        node.isFinal = w.isFinal;
        node.op = opOf ? opOf(w.stageKey) : BC::LifConv::Op::NHG;
        if(harmonicOf)
            node.n = harmonicOf(w.stageKey);
        nodes.push_back(std::move(node));
    }
    return nodes;
}

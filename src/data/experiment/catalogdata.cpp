#include "catalogdata.h"
#include <QSharedData>
#include <algorithm>

class CatalogDataPrivate : public QSharedData
{
public:
    QVector<TransitionData> transitions;
    QString sourceProgram;
    QString moleculeName;
    QVariantMap metadata;
    
    CatalogDataPrivate() = default;
    CatalogDataPrivate(const CatalogDataPrivate &other) = default;
};

CatalogData::CatalogData() : d(new CatalogDataPrivate)
{
}

CatalogData::CatalogData(const CatalogData &other) : d(other.d)
{
}

CatalogData &CatalogData::operator=(const CatalogData &other)
{
    if (this != &other)
        d.operator=(other.d);
    return *this;
}

CatalogData::~CatalogData() = default;

QVector<TransitionData> CatalogData::transitions() const
{
    return d->transitions;
}

void CatalogData::setTransitions(const QVector<TransitionData> &transitions)
{
    d->transitions = transitions;
}

void CatalogData::addTransition(const TransitionData &transition)
{
    d->transitions.append(transition);
}

void CatalogData::addTransition(double frequency, double intensity, const QString &quantumNumbers)
{
    d->transitions.append(TransitionData(frequency, intensity, quantumNumbers));
}

void CatalogData::clear()
{
    d->transitions.clear();
    d->sourceProgram.clear();
    d->moleculeName.clear();
    d->metadata.clear();
}

QString CatalogData::sourceProgram() const
{
    return d->sourceProgram;
}

void CatalogData::setSourceProgram(const QString &program)
{
    d->sourceProgram = program;
}

QString CatalogData::moleculeName() const
{
    return d->moleculeName;
}

void CatalogData::setMoleculeName(const QString &name)
{
    d->moleculeName = name;
}

QVariantMap CatalogData::metadata() const
{
    return d->metadata;
}

void CatalogData::setMetadata(const QVariantMap &metadata)
{
    d->metadata = metadata;
}

void CatalogData::setMetadataValue(const QString &key, const QVariant &value)
{
    d->metadata[key] = value;
}

QVariant CatalogData::metadataValue(const QString &key, const QVariant &defaultValue) const
{
    return d->metadata.value(key, defaultValue);
}

int CatalogData::size() const
{
    return d->transitions.size();
}

bool CatalogData::isEmpty() const
{
    return d->transitions.isEmpty();
}

TransitionData CatalogData::at(int index) const
{
    return d->transitions.at(index);
}

CatalogData CatalogData::filterByFrequencyRange(double minFreq, double maxFreq) const
{
    CatalogData filtered;
    filtered.setSourceProgram(d->sourceProgram);
    filtered.setMoleculeName(d->moleculeName);
    filtered.setMetadata(d->metadata);
    
    for (const auto &transition : d->transitions) {
        if (transition.frequency >= minFreq && transition.frequency <= maxFreq) {
            filtered.addTransition(transition);
        }
    }
    
    return filtered;
}

std::pair<double, double> CatalogData::frequencyRange() const
{
    if (d->transitions.isEmpty()) {
        return std::make_pair(0.0, 0.0);
    }
    
    auto minMax = std::minmax_element(d->transitions.begin(), d->transitions.end(),
                                      [](const TransitionData &a, const TransitionData &b) {
                                          return a.frequency < b.frequency;
                                      });
    
    return std::make_pair(minMax.first->frequency, minMax.second->frequency);
}
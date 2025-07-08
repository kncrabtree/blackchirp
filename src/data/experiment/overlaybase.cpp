#include "overlaybase.h"


OverlayBase::OverlayBase(OverlayType type) : d_type{type}
{
    
}

QVector<QPointF> OverlayBase::xyData() const
{
    // Return cached data if valid
    if (d_cacheValid) {
        return d_cachedFilteredData;
    }
    
    QVector<QPointF> rawData = _xyData();
    QVector<QPointF> transformedData;
    transformedData.reserve(rawData.size());
    
    // Apply scaling, offset transformations, and frequency filtering
    for (const QPointF& point : rawData) {
        double newX = point.x() + d_xOffset;
        double newY = (point.y() * d_yScale) + d_yOffset;
        
        // Apply frequency range filtering
        if (d_minFreqEnabled && newX < d_minFreqValue)
            continue;
        if (d_maxFreqEnabled && newX > d_maxFreqValue)
            continue;
            
        transformedData.append(QPointF(newX, newY));
    }
    
    // Cache the result
    d_cachedFilteredData = transformedData;
    d_cacheValid = true;
    
    return transformedData;
}

QString OverlayBase::getLabel() const
{
    return d_label;
}

QString OverlayBase::getSourceFile() const
{
    return d_sourceFile;
}

QString OverlayBase::getDestFile() const
{
    return d_destFile;
}

QString OverlayBase::getPlotId() const
{
    return d_plotId;
}

double OverlayBase::getYScale() const
{
    return d_yScale;
}

double OverlayBase::getYOffset() const
{
    return d_yOffset;
}

double OverlayBase::getXOffset() const
{
    return d_xOffset;
}

bool OverlayBase::getMinFreqEnabled() const
{
    return d_minFreqEnabled;
}

double OverlayBase::getMinFreqValue() const
{
    return d_minFreqValue;
}

bool OverlayBase::getMaxFreqEnabled() const
{
    return d_maxFreqEnabled;
}

double OverlayBase::getMaxFreqValue() const
{
    return d_maxFreqValue;
}

void OverlayBase::setLabel(const QString &newlabel)
{
    d_modified = true;
    d_label = newlabel;
}

void OverlayBase::setSourceFile(const QString &newsourceFile)
{
    d_modified = true;
    d_sourceFile = newsourceFile;
}

void OverlayBase::setDestFile(const QString &newdestFile)
{
    d_modified = true;
    d_destFile = newdestFile;
}

void OverlayBase::setPlotId(const QString &newplotId)
{
    d_modified = true;
    d_plotId = newplotId;
}

void OverlayBase::setYScale(double newyScale)
{
    d_modified = true;
    d_yScale = newyScale;
}

void OverlayBase::setYOffset(double newyOffset)
{
    d_modified = true;
    d_yOffset = newyOffset;
}

void OverlayBase::setXOffset(double newxOffset)
{
    d_modified = true;
    d_xOffset = newxOffset;
    invalidateCache();
}

void OverlayBase::setMinFreqLimit(bool enabled, double value)
{
    d_modified = true;
    d_minFreqEnabled = enabled;
    d_minFreqValue = value;
    invalidateCache();
}

void OverlayBase::setMaxFreqLimit(bool enabled, double value)
{
    d_modified = true;
    d_maxFreqEnabled = enabled;
    d_maxFreqValue = value;
    invalidateCache();
}

void OverlayBase::save()
{
    // Should these be here? Or implement a save function?
    writeToDest(); 
    d_modified = false;
}


void OverlayBase::storeMetadata(std::map<QString,QVariant> &m)
{
    using namespace BC::Key::Overlay;
    m.emplace(oLabel,d_label);
    m.emplace(oSourceFile,d_sourceFile);
    m.emplace(oDestFile,d_destFile);
    m.emplace(oPlotId,d_plotId);
    m.emplace(oYScale,d_yScale);
    m.emplace(oYOffset,d_yOffset);
    m.emplace(oXOffset,d_xOffset);
    m.emplace(oMinFreqEnabled,d_minFreqEnabled);
    m.emplace(oMinFreqValue,d_minFreqValue);
    m.emplace(oMaxFreqEnabled,d_maxFreqEnabled);
    m.emplace(oMaxFreqValue,d_maxFreqValue);
    
    // Add curve metadata with "curve_" prefix
    for(const auto& [key, value] : d_curveMetadata) {
        m.emplace("curve_" + key, value);
    }
    
    _storeMetadata(m);
    
}

void OverlayBase::retrieveMetadata(const std::map<QString,QVariant> &m)
{
    using namespace BC::Key::Overlay;
    
    auto it = m.find(oLabel);
    if(it != m.end())
        d_label = it->second.toString();
    it = m.find(oSourceFile);
    if(it != m.end())
        d_sourceFile = it->second.toString();
    it = m.find(oDestFile);
    if(it != m.end())
        d_destFile = it->second.toString();
    it = m.find(oPlotId);
    if(it != m.end())
        d_plotId = it->second.toString();
    it = m.find(oYScale);
    if(it != m.end())
        d_yScale = it->second.toDouble();
    it = m.find(oYOffset);
    if(it != m.end())
        d_yOffset = it->second.toDouble();
    it = m.find(oXOffset);
    if(it != m.end())
        d_xOffset = it->second.toDouble();
    it = m.find(oMinFreqEnabled);
    if(it != m.end())
        d_minFreqEnabled = it->second.toBool();
    it = m.find(oMinFreqValue);
    if(it != m.end())
        d_minFreqValue = it->second.toDouble();
    it = m.find(oMaxFreqEnabled);
    if(it != m.end())
        d_maxFreqEnabled = it->second.toBool();
    it = m.find(oMaxFreqValue);
    if(it != m.end())
        d_maxFreqValue = it->second.toDouble();
    
    // Invalidate cache after loading metadata
    invalidateCache();
    
    // Extract curve metadata (keys starting with "curve_")
    d_curveMetadata.clear();
    for(const auto& [key, value] : m) {
        if(key.startsWith("curve_")) {
            d_curveMetadata[key.mid(6)] = value;  // Remove "curve_" prefix
        }
    }
    
    _retrieveMetadata(m);
    
}

void OverlayBase::invalidateCache()
{
    d_cacheValid = false;
}

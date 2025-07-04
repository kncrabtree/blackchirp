#include "overlaybase.h"


OverlayBase::OverlayBase(OverlayType type) : d_type{type}
{
    
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
}

void OverlayBase::save()
{
    // Should these be here? Or implement a save function?
    writeToDest(); 
    d_modified = false;
}

void OverlayBase::loadFromSource()
{
    d_modified = true;   
    readFromSource();
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
    
    _retrieveMetadata(m);
    
}

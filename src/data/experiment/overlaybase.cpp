#include "overlaybase.h"


OverlayBase::OverlayBase(int num, QString path) :
    d_number{num}, d_path{path}
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

void OverlayBase::loadFromSource()
{
    d_modified = true;   
    readFromSource();
}

void OverlayBase::writeMetadata(std::map<QString, QVariant> &m)
{
    
}

void OverlayBase::readMetadata(const std::map<QString, QVariant> &m)
{
    
}

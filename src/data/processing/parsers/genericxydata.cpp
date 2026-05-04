#include "genericxydata.h"
#include <QFileInfo>

GenericXYData::GenericXYData()
{
    resetBounds();
}

void GenericXYData::setData(const QVector<QPointF> &data)
{
    d_data = data;
    resetBounds();
    
    // Recalculate bounds for all data points
    for (const QPointF &point : d_data) {
        updateBounds(point);
    }
}

void GenericXYData::addDataPoint(const QPointF &point)
{
    if (!point.isNull()) {
        d_data.append(point);
        updateBounds(point);
    }
}

void GenericXYData::clear()
{
    d_data.clear();
    d_filePath.clear();
    d_columnNames.clear();
    d_xColumn = 0;
    d_yColumn = 1;
    d_delimiter = ",";
    d_headerLines = 0;
    d_hasColumnHeaders = false;
    d_totalLines = 0;
    d_errorMessage.clear();
    resetBounds();
}

void GenericXYData::updateBounds(const QPointF &point)
{
    double x = point.x();
    double y = point.y();
    
    if (qIsNaN(d_xMin) || x < d_xMin) d_xMin = x;
    if (qIsNaN(d_xMax) || x > d_xMax) d_xMax = x;
    if (qIsNaN(d_yMin) || y < d_yMin) d_yMin = y;
    if (qIsNaN(d_yMax) || y > d_yMax) d_yMax = y;
}

void GenericXYData::resetBounds()
{
    d_xMin = qQNaN();
    d_xMax = qQNaN();
    d_yMin = qQNaN();
    d_yMax = qQNaN();
}
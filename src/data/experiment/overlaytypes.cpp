#include "overlaytypes.h"

#include <data/storage/blackchirpcsv.h>
#include <data/experiment/experiment.h>


BCExpOverlay::BCExpOverlay() :
    OverlayBase(BCExperiment)
{

}


QVector<QPointF> BCExpOverlay::_xyData() const
{
    return d_ft.toVector();
}

void BCExpOverlay::setFtData(const Ft &ftData)
{
    d_ft = ftData;
    setModified(true);
}

void BCExpOverlay::readFromDest()
{
    QString destFile = getDestFile();
    if(destFile.isEmpty())
        return;

    QFile f(destFile);
    if(!f.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QVector<double> ftData;

    // Skip the header line if present
    auto headerLine = f.readLine().trimmed();

    // Read the y-values line by line
    while(!f.atEnd())
    {
        auto line = f.readLine().trimmed();
        if(line.isEmpty())
            continue;

        bool ok = false;
        double value = line.toDouble(&ok);
        if(ok)
            ftData.append(value);
    }

    f.close();

    // Calculate yMin and yMax from the loaded data
    double yMin = 0.0, yMax = 0.0;
    if(!ftData.isEmpty())
    {
        yMin = yMax = ftData.constFirst();
        for(const auto &val : ftData)
        {
            yMin = qMin(yMin, val);
            yMax = qMax(yMax, val);
        }
    }

    // Set the data in the Ft object (this also sets yMin/yMax)
    d_ft.setData(ftData, yMin, yMax);
}

void BCExpOverlay::writeToDest()
{
    QString destFile = getDestFile();
    if(destFile.isEmpty())
        return;

    QFile f(destFile);

    // Use BlackchirpCSV::writeY template function to write the Y data
    if(!BlackchirpCSV::writeY(f, d_ft.yData(), QString("FT Magnitude")))
    {
        // Handle error if needed - file writing failed
        return;
    }
}

void BCExpOverlay::_storeMetadata(std::map<QString, QVariant> &m)
{
    using namespace BC::Key::Overlay;
    m.emplace(ftYMin, d_ft.yMin());
    m.emplace(ftYMax, d_ft.yMax());
    m.emplace(ftX0MHz, d_ft.xFirst());
    m.emplace(ftSpacingMHz, d_ft.xSpacing());
    m.emplace(ftLoFreqMHz, d_ft.loFreqMHz());
    m.emplace(ftShots, static_cast<qulonglong>(d_ft.shots()));
}

void BCExpOverlay::_retrieveMetadata(const std::map<QString, QVariant> &m)
{
    using namespace BC::Key::Overlay;

    auto it = m.find(ftYMin);
    if(it != m.end()) {
        // Note: yMin/yMax are typically set via setData() or during FT processing
        // They may need to be reconstructed from the actual data when loaded
    }

    it = m.find(ftYMax);
    if(it != m.end()) {
        // Note: yMin/yMax are typically set via setData() or during FT processing
    }

    it = m.find(ftX0MHz);
    if(it != m.end())
        d_ft.setX0(it->second.toDouble());

    it = m.find(ftSpacingMHz);
    if(it != m.end())
        d_ft.setSpacing(it->second.toDouble());

    it = m.find(ftLoFreqMHz);
    if(it != m.end())
        d_ft.setLoFreq(it->second.toDouble());

    it = m.find(ftShots);
    if(it != m.end())
        d_ft.setNumShots(it->second.toULongLong());
}

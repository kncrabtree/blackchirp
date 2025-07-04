#include "overlaytypes.h"

#include <data/storage/blackchirpcsv.h>
#include <data/experiment/experiment.h>


BCExpOverlay::BCExpOverlay(int experimentNumber, const QString &experimentPath, int frame) :
    OverlayBase(BCExperiment), d_frame{frame}, d_experimentNumber{experimentNumber}, d_experimentPath{experimentPath}
{

}


QVector<QPointF> BCExpOverlay::xyData() const
{
    return d_ft.toVector();
}

void BCExpOverlay::readFromSource()
{
    d_errorString.clear();

    // Load the experiment using the stored number and path
    auto experiment = std::make_shared<Experiment>(d_experimentNumber, d_experimentPath);

    // Check if FTMW is enabled
    if(!experiment->ftmwEnabled())
    {
        d_errorString = "Source experiment does not contain FTMW data";
        return;
    }

    // Get the FID storage
    auto fidStorage = experiment->ftmwConfig()->storage();
    if(!fidStorage)
    {
        d_errorString = "Could not access FID storage from experiment";
        return;
    }

    // Determine processing settings
    FtWorker::FidProcessingSettings settings;

    if(d_useAutomaticProcessing)
    {
        // Use automatic processing - try to read from processing.csv file first
        if(!fidStorage->readProcessingSettings(settings))
        {
            // Use default settings if no file exists
            settings.startUs = 5.0;
            settings.endUs = 10.0;
            settings.expFilter = 0.0;
            settings.zeroPadFactor = 0;
            settings.removeDC = true;
            settings.units = FtWorker::FtuV;
            settings.autoScaleIgnoreMHz = 250.0;
            settings.windowFunction = FtWorker::None;
        }
    }
    else
    {
        // Use user-specified processing settings
        settings = d_processingSettings;
    }

    // Get the FID list
    FidList fidList = fidStorage->getCurrentFidList();
    if(fidList.isEmpty())
    {
        d_errorString = "No FID data found in source experiment";
        return;
    }

    // Create FtWorker and process the FIDs synchronously
    FtWorker worker;
    Ft result = worker.doFT(fidList, settings, d_frame, -1, false);

    if(result.isEmpty())
    {
        d_errorString = "FT processing failed or returned empty result";
        return;
    }

    // Store the result
    d_ft = result;
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

    // Store frame selection
    m.emplace(frame, d_frame);

    // Store processing settings if they've been set by user
    if(d_processingSettings.startUs > 0.0 || d_processingSettings.endUs > 0.0)
    {
        m.emplace(procStartUs, d_processingSettings.startUs);
        m.emplace(procEndUs, d_processingSettings.endUs);
        m.emplace(procExpFilter, d_processingSettings.expFilter);
        m.emplace(procZeroPadFactor, d_processingSettings.zeroPadFactor);
        m.emplace(procRemoveDC, d_processingSettings.removeDC);
        m.emplace(procUnits, static_cast<int>(d_processingSettings.units));
        m.emplace(procAutoScaleIgnoreMHz, d_processingSettings.autoScaleIgnoreMHz);
        m.emplace(procWindowFunction, static_cast<int>(d_processingSettings.windowFunction));
    }
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

    // Retrieve frame selection
    it = m.find(frame);
    if(it != m.end())
        d_frame = it->second.toInt();

    // Retrieve processing settings if present
    it = m.find(procStartUs);
    if(it != m.end())
    {
        d_processingSettings.startUs = it->second.toDouble();

        // If we found one processing setting, load them all
        auto it2 = m.find(procEndUs);
        if(it2 != m.end()) d_processingSettings.endUs = it2->second.toDouble();

        it2 = m.find(procExpFilter);
        if(it2 != m.end()) d_processingSettings.expFilter = it2->second.toDouble();

        it2 = m.find(procZeroPadFactor);
        if(it2 != m.end()) d_processingSettings.zeroPadFactor = it2->second.toInt();

        it2 = m.find(procRemoveDC);
        if(it2 != m.end()) d_processingSettings.removeDC = it2->second.toBool();

        it2 = m.find(procUnits);
        if(it2 != m.end()) d_processingSettings.units = static_cast<FtWorker::FtUnits>(it2->second.toInt());

        it2 = m.find(procAutoScaleIgnoreMHz);
        if(it2 != m.end()) d_processingSettings.autoScaleIgnoreMHz = it2->second.toDouble();

        it2 = m.find(procWindowFunction);
        if(it2 != m.end()) d_processingSettings.windowFunction = static_cast<FtWorker::FtWindowFunction>(it2->second.toInt());
    }
}

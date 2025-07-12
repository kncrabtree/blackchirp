#include "overlaytypes.h"

#include <data/storage/blackchirpcsv.h>
#include <data/experiment/experiment.h>
#include <QtMath>
#include <cmath>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>


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

Ft BCExpOverlay::getFtData() const
{
    return d_ft;
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

// CatalogOverlay implementation

CatalogOverlay::CatalogOverlay() : OverlayBase(Catalog)
{
}

CatalogData CatalogOverlay::catalogData() const
{
    return d_catalogData;
}

void CatalogOverlay::setCatalogData(const CatalogData &data)
{
    d_catalogData = data;
    invalidateConvolutionCache();
    setModified(true);
}

bool CatalogOverlay::convolutionEnabled() const
{
    return d_convolutionEnabled;
}

void CatalogOverlay::setConvolutionEnabled(bool enabled)
{
    if (d_convolutionEnabled != enabled) {
        d_convolutionEnabled = enabled;
        invalidateConvolutionCache();
        setModified(true);
    }
}

CatalogOverlay::LineshapeType CatalogOverlay::lineshapeType() const
{
    return d_lineshapeType;
}

void CatalogOverlay::setLineshapeType(LineshapeType type)
{
    if (d_lineshapeType != type) {
        d_lineshapeType = type;
        invalidateConvolutionCache();
        setModified(true);
    }
}

double CatalogOverlay::linewidth() const
{
    return d_linewidth;
}

void CatalogOverlay::setLinewidth(double width)
{
    if (qAbs(d_linewidth - width) > 1e-6) {
        d_linewidth = width;
        invalidateConvolutionCache();
        setModified(true);
    }
}

double CatalogOverlay::convolutionMinFreq() const
{
    return d_convolutionMinFreq;
}

double CatalogOverlay::convolutionMaxFreq() const
{
    return d_convolutionMaxFreq;
}

void CatalogOverlay::setConvolutionFreqRange(double minFreq, double maxFreq)
{
    if (qAbs(d_convolutionMinFreq - minFreq) > 1e-6 || 
        qAbs(d_convolutionMaxFreq - maxFreq) > 1e-6) {
        d_convolutionMinFreq = minFreq;
        d_convolutionMaxFreq = maxFreq;
        invalidateConvolutionCache();
        setModified(true);
    }
}

double CatalogOverlay::pointSpacing() const
{
    return d_pointSpacing;
}

void CatalogOverlay::setPointSpacing(double spacing)
{
    if (qAbs(d_pointSpacing - spacing) > 1e-9) {
        d_pointSpacing = spacing;
        invalidateConvolutionCache();
        setModified(true);
    }
}

void CatalogOverlay::setConvolutionSettings(bool enabled, LineshapeType lineshape, 
                                           double linewidth, double minFreq, double maxFreq, 
                                           double spacing)
{
    d_convolutionEnabled = enabled;
    d_lineshapeType = lineshape;
    d_linewidth = linewidth;
    d_convolutionMinFreq = minFreq;
    d_convolutionMaxFreq = maxFreq;
    d_pointSpacing = spacing;
    invalidateConvolutionCache();
    setModified(true);
}

QVector<QPointF> CatalogOverlay::_xyData() const
{
    if (d_convolutionEnabled) {
        if (!d_convolutionCacheValid) {
            d_convolvedCache = generateConvolvedSpectrum();
            d_convolutionCacheValid = true;
        }
        return d_convolvedCache;
    } else {
        // Return raw transition data as stick spectrum
        QVector<QPointF> transitions;
        transitions.reserve(d_catalogData.size());
        
        for (int i = 0; i < d_catalogData.size(); ++i) {
            const TransitionData &trans = d_catalogData.at(i);
            transitions.append(QPointF(trans.frequency, trans.intensity));
        }
        
        return transitions;
    }
}

QVector<QPointF> CatalogOverlay::generateConvolvedSpectrum() const
{
    if (d_catalogData.isEmpty()) {
        return QVector<QPointF>();
    }
    
    // Generate frequency grid
    int numPoints = static_cast<int>((d_convolutionMaxFreq - d_convolutionMinFreq) / d_pointSpacing) + 1;
    QVector<QPointF> spectrum;
    spectrum.reserve(numPoints);
    
    // Initialize spectrum
    for (int i = 0; i < numPoints; ++i) {
        double freq = d_convolutionMinFreq + i * d_pointSpacing;
        spectrum.append(QPointF(freq, 0.0));
    }
    
    // Convolve each transition
    for (int i = 0; i < d_catalogData.size(); ++i) {
        const TransitionData &trans = d_catalogData.at(i);
        
        // Skip transitions outside frequency range
        if (trans.frequency < d_convolutionMinFreq || trans.frequency > d_convolutionMaxFreq) {
            continue;
        }
        
        // Add contribution to each grid point
        for (int j = 0; j < spectrum.size(); ++j) {
            double gridFreq = spectrum[j].x();
            double contribution = 0.0;
            
            if (d_lineshapeType == Lorentzian) {
                contribution = lorentzianProfile(gridFreq, trans.frequency, d_linewidth);
            } else {
                contribution = gaussianProfile(gridFreq, trans.frequency, d_linewidth);
            }
            
            spectrum[j].setY(spectrum[j].y() + trans.intensity * contribution);
        }
    }
    
    return spectrum;
}

double CatalogOverlay::lorentzianProfile(double x, double x0, double fwhmKHz) const
{
    // Convert kHz FWHM to MHz for calculation
    double fwhmMHz = fwhmKHz / 1000.0;
    double gamma = fwhmMHz / 2.0;  // Half-width at half-maximum
    
    double dx = x - x0;
    return (gamma / M_PI) / (dx * dx + gamma * gamma);
}

double CatalogOverlay::gaussianProfile(double x, double x0, double fwhmKHz) const
{
    // Convert kHz FWHM to MHz for calculation
    double fwhmMHz = fwhmKHz / 1000.0;
    double sigma = fwhmMHz / (2.0 * sqrt(2.0 * log(2.0)));  // Convert FWHM to sigma
    
    double dx = x - x0;
    return (1.0 / (sigma * sqrt(2.0 * M_PI))) * exp(-0.5 * (dx / sigma) * (dx / sigma));
}

void CatalogOverlay::invalidateConvolutionCache()
{
    d_convolutionCacheValid = false;
}

void CatalogOverlay::readFromDest()
{
    QString destFile = getDestFile();
    if(destFile.isEmpty())
        return;

    QFile f(destFile);
    if(!f.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QTextStream stream(&f);
    
    // Skip header line
    if(!stream.atEnd())
        stream.readLine();
    
    // Read transition data
    QVector<TransitionData> transitions;
    
    while(!stream.atEnd()) {
        QString line = stream.readLine().trimmed();
        if(line.isEmpty())
            continue;
            
        QStringList parts = line.split(BC::CSV::del);
        if(parts.size() < 3)
            continue;
            
        TransitionData trans;
        bool ok;
        trans.frequency = parts[0].toDouble(&ok);
        if(!ok) continue;
        
        trans.intensity = parts[1].toDouble(&ok);
        if(!ok) continue;
        
        trans.quantumNumbers = parts[2];
        
        // Parse additional data if present (JSON format)
        if(parts.size() > 3 && !parts[3].isEmpty()) {
            QJsonParseError parseError;
            QJsonDocument doc = QJsonDocument::fromJson(parts[3].toUtf8(), &parseError);
            if(parseError.error == QJsonParseError::NoError && doc.isObject()) {
                QJsonObject obj = doc.object();
                for(auto it = obj.begin(); it != obj.end(); ++it) {
                    trans.additionalData.insert(it.key(), it.value().toVariant());
                }
            }
        }
        
        transitions.append(trans);
    }
    
    f.close();
    
    // Create CatalogData and set it
    CatalogData data;
    data.setTransitions(transitions);
    setCatalogData(data);
}

void CatalogOverlay::writeToDest()
{
    QString destFile = getDestFile();
    if(destFile.isEmpty() || d_catalogData.isEmpty())
        return;

    QFile f(destFile);
    
    // Prepare data vectors for BlackchirpCSV using QVariant for automatic formatting
    QVector<QVariant> frequencies, intensities, quantumNumbers, additionalData;
    
    frequencies.reserve(d_catalogData.size());
    intensities.reserve(d_catalogData.size());
    quantumNumbers.reserve(d_catalogData.size());
    additionalData.reserve(d_catalogData.size());
    
    for(int i = 0; i < d_catalogData.size(); ++i) {
        const TransitionData &trans = d_catalogData.at(i);
        frequencies.append(trans.frequency);
        intensities.append(trans.intensity);
        quantumNumbers.append(trans.quantumNumbers);
        
        // Convert additional data to JSON string (semicolons already removed by parser)
        if(!trans.additionalData.isEmpty()) {
            QJsonObject obj;
            for(auto it = trans.additionalData.begin(); it != trans.additionalData.end(); ++it) {
                obj.insert(it.key(), QJsonValue::fromVariant(it.value()));
            }
            QJsonDocument doc(obj);
            additionalData.append(QString::fromUtf8(doc.toJson(QJsonDocument::Compact)));
        } else {
            additionalData.append(QString());
        }
    }
    
    // Use BlackchirpCSV to write the data
    if(!BlackchirpCSV::writeYMultiple(f, 
                                     {"Frequency(MHz)", "Intensity", "QuantumNumbers", "AdditionalData"},
                                     {frequencies, intensities, quantumNumbers, additionalData})) {
        // Handle error if needed
        return;
    }
}

void CatalogOverlay::_storeMetadata(std::map<QString, QVariant> &m)
{
    using namespace BC::Key::Overlay::Catalog;
    
    m.emplace(sourceProgram, d_catalogData.sourceProgram());
    m.emplace(moleculeName, d_catalogData.moleculeName());
    m.emplace(BC::Key::Overlay::Catalog::convolutionEnabled, d_convolutionEnabled);
    m.emplace(BC::Key::Overlay::Catalog::lineshapeType, static_cast<int>(d_lineshapeType));
    m.emplace(linewidthKHz, d_linewidth);
    m.emplace(BC::Key::Overlay::Catalog::convolutionMinFreq, d_convolutionMinFreq);
    m.emplace(BC::Key::Overlay::Catalog::convolutionMaxFreq, d_convolutionMaxFreq);
    m.emplace(BC::Key::Overlay::Catalog::pointSpacing, d_pointSpacing);
    m.emplace(transitionCount, d_catalogData.size());
    
    // Store frequency range
    if (!d_catalogData.isEmpty()) {
        auto range = d_catalogData.frequencyRange();
        QString rangeStr = QString("%1-%2").arg(range.first).arg(range.second);
        m.emplace(frequencyRange, rangeStr);
    }
}

void CatalogOverlay::_retrieveMetadata(const std::map<QString, QVariant> &m)
{
    using namespace BC::Key::Overlay::Catalog;
    
    auto it = m.find(sourceProgram);
    if (it != m.end()) {
        d_catalogData.setSourceProgram(it->second.toString());
    }
    
    it = m.find(moleculeName);
    if (it != m.end()) {
        d_catalogData.setMoleculeName(it->second.toString());
    }
    
    it = m.find(BC::Key::Overlay::Catalog::convolutionEnabled);
    if (it != m.end()) {
        d_convolutionEnabled = it->second.toBool();
    }
    
    it = m.find(BC::Key::Overlay::Catalog::lineshapeType);
    if (it != m.end()) {
        d_lineshapeType = static_cast<LineshapeType>(it->second.toInt());
    }
    
    it = m.find(linewidthKHz);
    if (it != m.end()) {
        d_linewidth = it->second.toDouble();
    }
    
    it = m.find(BC::Key::Overlay::Catalog::convolutionMinFreq);
    if (it != m.end()) {
        d_convolutionMinFreq = it->second.toDouble();
    }
    
    it = m.find(BC::Key::Overlay::Catalog::convolutionMaxFreq);
    if (it != m.end()) {
        d_convolutionMaxFreq = it->second.toDouble();
    }
    
    it = m.find(BC::Key::Overlay::Catalog::pointSpacing);
    if (it != m.end()) {
        d_pointSpacing = it->second.toDouble();
    }
    
    // Invalidate cache after loading metadata
    invalidateConvolutionCache();
}

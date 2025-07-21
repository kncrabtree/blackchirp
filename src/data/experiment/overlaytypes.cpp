#include "overlaytypes.h"

#include <data/storage/blackchirpcsv.h>
#include <data/experiment/experiment.h>
#include <gui/plot/blackchirpplotcurve.h>
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
    if (d_catalogData != data) {
        d_catalogData = data;
        invalidateConvolutionCache();
        setModified(true);
    }
}

bool CatalogOverlay::convolutionEnabled() const
{
    return d_convolutionEnabled;
}

void CatalogOverlay::setConvolutionEnabled(bool enabled)
{
    if (d_convolutionEnabled != enabled) {
        d_convolutionEnabled = enabled;
        
        // Only invalidate cache if enabling convolution and no cached data exists
        if (enabled && !hasConvolvedData()) {
            invalidateConvolutionCache();
        }
        
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

int CatalogOverlay::numConvolutionPoints() const
{
    return d_numConvolutionPoints;
}

void CatalogOverlay::setNumConvolutionPoints(int numPoints)
{
    if (d_numConvolutionPoints != numPoints) {
        d_numConvolutionPoints = numPoints;
        invalidateConvolutionCache();
        setModified(true);
    }
}

double CatalogOverlay::calculatePointSpacing() const
{
    if (d_numConvolutionPoints <= 1) {
        return d_convolutionMaxFreq - d_convolutionMinFreq;
    }
    return (d_convolutionMaxFreq - d_convolutionMinFreq) / (d_numConvolutionPoints - 1);
}


double CatalogOverlay::filterMinFreq() const
{
    return d_filterMinFreq;
}

double CatalogOverlay::filterMaxFreq() const
{
    return d_filterMaxFreq;
}

void CatalogOverlay::setFilterRange(double minFreq, double maxFreq)
{
    if (qAbs(d_filterMinFreq - minFreq) > 1e-6 || 
        qAbs(d_filterMaxFreq - maxFreq) > 1e-6) {
        d_filterMinFreq = minFreq;
        d_filterMaxFreq = maxFreq;
        setModified(true);
    }
}

void CatalogOverlay::setConvolutionSettings(bool enabled, LineshapeType lineshape, 
                                           double linewidth, double minFreq, double maxFreq, 
                                           int numPoints)
{
    d_convolutionEnabled = enabled;
    d_lineshapeType = lineshape;
    d_linewidth = linewidth;
    d_convolutionMinFreq = minFreq;
    d_convolutionMaxFreq = maxFreq;
    d_numConvolutionPoints = numPoints;
    invalidateConvolutionCache();
    setModified(true);
}

QVector<QPointF> CatalogOverlay::_xyData() const
{
    if (d_convolutionEnabled) {
        switch (d_cacheState) {
        case CacheState::Valid:
            return d_convolvedCache;
            
        case CacheState::Pending:
            // Background operation in progress - return previous cache or fall through to raw data
            if (!d_convolvedCache.isEmpty()) {
                return d_convolvedCache; // Return stale data while updating
            }
            // Fall through to return raw data as placeholder
            
        case CacheState::Invalid:
            // Cache invalid - fall through to return raw data as placeholder
            break;
        }
    }
    
    // Return raw transition data as stick spectrum (used for non-convolved mode and as placeholder)
    QVector<QPointF> transitions;
    transitions.reserve(d_catalogData.size());
    
    for (int i = 0; i < d_catalogData.size(); ++i) {
        const TransitionData &trans = d_catalogData.at(i);
        transitions.append(QPointF(trans.frequency, trans.intensity));
    }
    
    return transitions;
}

QVector<QPointF> CatalogOverlay::generateConvolvedSpectrum() const
{
    if (d_catalogData.isEmpty()) {
        return QVector<QPointF>();
    }

    //pre-filter transitions outside range; place into lightweight structures
    QVector<double> x0, y0;
    x0.reserve(d_catalogData.size());
    y0.reserve(d_catalogData.size());
    for(const auto &trans : d_catalogData.transitions())
    {
        if (trans.frequency >= d_convolutionMinFreq && trans.frequency <= d_convolutionMaxFreq)
        {
            x0.append(trans.frequency);
            y0.append(trans.intensity);
        }
    }
    
    // Generate frequency grid using number of points
    double pointSpacing = calculatePointSpacing();
    QVector<QPointF> spectrum;
    spectrum.reserve(d_numConvolutionPoints);

    //Store lineshape function pointer
    auto f = &CatalogOverlay::lorentzianProfile;
    if(d_lineshapeType == Gaussian)
        f = &CatalogOverlay::gaussianProfile;
    
    
    // Add contribution to each grid point
    for (int i = 0; i < d_numConvolutionPoints; ++i) {
        double yy = 0.0;
        double gridFreq = d_convolutionMinFreq + i * pointSpacing;
        for (int j = 0; (j < x0.size()) && (j < y0.size()); ++j) {
            yy += y0.at(j) * (this->*f)(gridFreq, x0.at(j), d_linewidth);
        }
        spectrum.append({gridFreq,yy});
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
    d_cacheState = CacheState::Invalid;
    // Invalidate base class cache to force refresh from _xyData()
    invalidateCache();
}

void CatalogOverlay::setCachePending()
{
    d_cacheState = CacheState::Pending;
    // Invalidate base class cache to force refresh from _xyData()
    invalidateCache();
}

void CatalogOverlay::setCacheValid(const QVector<QPointF> &convolvedData)
{
    d_convolvedCache = convolvedData;
    d_cacheState = CacheState::Valid;
    // Invalidate base class cache to force refresh from _xyData()
    invalidateCache();
}

bool CatalogOverlay::isCacheValid() const
{
    return d_cacheState == CacheState::Valid;
}

bool CatalogOverlay::hasConvolvedData() const
{
    return d_cacheState == CacheState::Valid && !d_convolvedCache.isEmpty();
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
    m.emplace(BC::Key::Overlay::Catalog::numConvolutionPoints, d_numConvolutionPoints);
    m.emplace(transitionCount, d_catalogData.size());
    
    // Store filtering range settings
    m.emplace(BC::Key::Overlay::Catalog::filterMinFreq, d_filterMinFreq);
    m.emplace(BC::Key::Overlay::Catalog::filterMaxFreq, d_filterMaxFreq);
    
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
        // Force convolution to disabled when loading from disk
        d_convolutionEnabled = false;
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
    
    it = m.find(BC::Key::Overlay::Catalog::numConvolutionPoints);
    if (it != m.end()) {
        d_numConvolutionPoints = it->second.toInt();
    }
    
    // Retrieve filtering range settings
    it = m.find(BC::Key::Overlay::Catalog::filterMinFreq);
    if (it != m.end()) {
        d_filterMinFreq = it->second.toDouble();
    }
    
    it = m.find(BC::Key::Overlay::Catalog::filterMaxFreq);
    if (it != m.end()) {
        d_filterMaxFreq = it->second.toDouble();
    }
    
    // Force curve style to stick plot when loaded from disk
    setCurveMetadata(BC::Key::bcCurveCurveStyle, static_cast<int>(QwtPlotCurve::Sticks));
    
    // Invalidate cache after loading metadata
    invalidateConvolutionCache();
}

// GenericXYOverlay implementation

GenericXYOverlay::GenericXYOverlay() : OverlayBase(GenericXY)
{
}

QVector<QPointF> GenericXYOverlay::rawData() const
{
    return d_rawData;
}

void GenericXYOverlay::setRawData(const QVector<QPointF> &data)
{
    if (d_rawData != data) {
        d_rawData = data;
        updateStatistics();
        setModified(true);
    }
}

QString GenericXYOverlay::delimiter() const
{
    return d_delimiter;
}

void GenericXYOverlay::setDelimiter(const QString &delim)
{
    if (d_delimiter != delim) {
        d_delimiter = delim;
        setModified(true);
    }
}

int GenericXYOverlay::headerLines() const
{
    return d_headerLines;
}

void GenericXYOverlay::setHeaderLines(int lines)
{
    if (d_headerLines != lines) {
        d_headerLines = qMax(0, lines);
        setModified(true);
    }
}

int GenericXYOverlay::xColumn() const
{
    return d_xColumn;
}

int GenericXYOverlay::yColumn() const
{
    return d_yColumn;
}

void GenericXYOverlay::setDataColumns(int xCol, int yCol)
{
    if (d_xColumn != xCol || d_yColumn != yCol) {
        d_xColumn = qMax(0, xCol);
        d_yColumn = qMax(0, yCol);
        setModified(true);
    }
}

QStringList GenericXYOverlay::columnNames() const
{
    return d_columnNames;
}

void GenericXYOverlay::setColumnNames(const QStringList &names)
{
    if (d_columnNames != names) {
        d_columnNames = names;
        setModified(true);
    }
}

int GenericXYOverlay::dataPointCount() const
{
    return d_dataPoints;
}

double GenericXYOverlay::xMin() const
{
    return d_xMin;
}

double GenericXYOverlay::xMax() const
{
    return d_xMax;
}

double GenericXYOverlay::yMin() const
{
    return d_yMin;
}

double GenericXYOverlay::yMax() const
{
    return d_yMax;
}

QPair<double, double> GenericXYOverlay::xRange() const
{
    return qMakePair(d_xMin, d_xMax);
}

QPair<double, double> GenericXYOverlay::yRange() const
{
    return qMakePair(d_yMin, d_yMax);
}

double GenericXYOverlay::filterMinX() const
{
    return d_filterMinX;
}

double GenericXYOverlay::filterMaxX() const
{
    return d_filterMaxX;
}

void GenericXYOverlay::setFilterRange(double minX, double maxX)
{
    if (d_filterMinX != minX || d_filterMaxX != maxX) {
        d_filterMinX = minX;
        d_filterMaxX = maxX;
        setModified();
    }
}

QVector<QPointF> GenericXYOverlay::_xyData() const
{
    return d_rawData;
}

void GenericXYOverlay::updateStatistics()
{
    d_dataPoints = d_rawData.size();
    
    if (d_rawData.isEmpty()) {
        d_xMin = d_xMax = d_yMin = d_yMax = 0.0;
        return;
    }
    
    // Initialize with first point
    const QPointF &first = d_rawData.constFirst();
    d_xMin = d_xMax = first.x();
    d_yMin = d_yMax = first.y();
    
    // Find min/max values
    for (const QPointF &point : d_rawData) {
        d_xMin = qMin(d_xMin, point.x());
        d_xMax = qMax(d_xMax, point.x());
        d_yMin = qMin(d_yMin, point.y());
        d_yMax = qMax(d_yMax, point.y());
    }
}

GenericXYOverlay::DelimiterType GenericXYOverlay::stringToDelimiterType(const QString &delimiter) const
{
    if (delimiter == ",") return DelimiterType::Comma;
    if (delimiter == "\t") return DelimiterType::Tab;
    if (delimiter == " ") return DelimiterType::Space;
    if (delimiter == ";") return DelimiterType::Semicolon;
    if (delimiter.trimmed().isEmpty() && delimiter.contains(QRegularExpression("\\s+"))) return DelimiterType::Whitespace;
    
    // Default to comma for unknown delimiters
    return DelimiterType::Comma;
}

QString GenericXYOverlay::delimiterTypeToString(GenericXYOverlay::DelimiterType type) const
{
    switch (type) {
    case DelimiterType::Comma:     return ",";
    case DelimiterType::Tab:       return "\t";
    case DelimiterType::Space:     return " ";
    case DelimiterType::Semicolon: return ";";
    case DelimiterType::Whitespace: return " "; // Default to single space for whitespace
    }
    
    return ","; // Default fallback
}

void GenericXYOverlay::readFromDest()
{
    QString destFile = getDestFile();
    if (destFile.isEmpty())
        return;

    QFile f(destFile);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QTextStream stream(&f);
    
    // Skip header line
    if (!stream.atEnd())
        stream.readLine();
    
    // Read XY data
    QVector<QPointF> data;
    
    while (!stream.atEnd()) {
        QString line = stream.readLine().trimmed();
        if (line.isEmpty())
            continue;
            
        QStringList parts = line.split(BC::CSV::del);
        if (parts.size() < 2)
            continue;
            
        bool xOk, yOk;
        double x = parts[0].toDouble(&xOk);
        double y = parts[1].toDouble(&yOk);
        
        if (xOk && yOk) {
            data.append(QPointF(x, y));
        }
    }
    
    f.close();
    setRawData(data);
}

void GenericXYOverlay::writeToDest()
{
    QString destFile = getDestFile();
    if (destFile.isEmpty() || d_rawData.isEmpty())
        return;

    QFile f(destFile);
    
    // Prepare data vectors for BlackchirpCSV
    QVector<QVariant> xData, yData;
    xData.reserve(d_rawData.size());
    yData.reserve(d_rawData.size());
    
    for (const QPointF &point : d_rawData) {
        xData.append(point.x());
        yData.append(point.y());
    }
    
    // Use BlackchirpCSV to write the XY data
    if (!BlackchirpCSV::writeYMultiple(f, 
                                      {"X", "Y"},
                                      {xData, yData})) {
        // Handle error if needed
        return;
    }
}

void GenericXYOverlay::_storeMetadata(std::map<QString, QVariant> &m)
{
    using namespace BC::Key::Overlay::GenericXY;
    
    // Store delimiter as enum to avoid BlackchirpCSV conflicts
    m.emplace(BC::Key::Overlay::GenericXY::delimiter, static_cast<int>(stringToDelimiterType(d_delimiter)));
    m.emplace(BC::Key::Overlay::GenericXY::headerLines, d_headerLines);
    m.emplace(BC::Key::Overlay::GenericXY::xColumn, d_xColumn);
    m.emplace(BC::Key::Overlay::GenericXY::yColumn, d_yColumn);
    // Serialize QStringList manually for BlackchirpCSV compatibility
    QString serializedColumnNames = d_columnNames.join(BC::CSV::altDel);
    m.emplace(BC::Key::Overlay::GenericXY::columnNames, serializedColumnNames);
    m.emplace(BC::Key::Overlay::GenericXY::dataPoints, d_dataPoints);
    m.emplace(BC::Key::Overlay::GenericXY::xMin, d_xMin);
    m.emplace(BC::Key::Overlay::GenericXY::xMax, d_xMax);
    m.emplace(BC::Key::Overlay::GenericXY::yMin, d_yMin);
    m.emplace(BC::Key::Overlay::GenericXY::yMax, d_yMax);
    m.emplace(BC::Key::Overlay::GenericXY::filterMinX, d_filterMinX);
    m.emplace(BC::Key::Overlay::GenericXY::filterMaxX, d_filterMaxX);
}

void GenericXYOverlay::_retrieveMetadata(const std::map<QString, QVariant> &m)
{
    using namespace BC::Key::Overlay::GenericXY;
    
    auto it = m.find(BC::Key::Overlay::GenericXY::delimiter);
    if (it != m.end()) {
        DelimiterType delimiterType = static_cast<DelimiterType>(it->second.toInt());
        d_delimiter = delimiterTypeToString(delimiterType);
    }
    
    it = m.find(BC::Key::Overlay::GenericXY::headerLines);
    if (it != m.end()) {
        d_headerLines = it->second.toInt();
    }
    
    it = m.find(BC::Key::Overlay::GenericXY::xColumn);
    if (it != m.end()) {
        d_xColumn = it->second.toInt();
    }
    
    it = m.find(BC::Key::Overlay::GenericXY::yColumn);
    if (it != m.end()) {
        d_yColumn = it->second.toInt();
    }
    
    it = m.find(BC::Key::Overlay::GenericXY::columnNames);
    if (it != m.end()) {
        // Deserialize manually serialized QStringList
        QString serializedColumnNames = it->second.toString();
        if (!serializedColumnNames.isEmpty()) {
            d_columnNames = serializedColumnNames.split(BC::CSV::altDel);
        } else {
            d_columnNames.clear();
        }
    }
    
    it = m.find(BC::Key::Overlay::GenericXY::dataPoints);
    if (it != m.end()) {
        d_dataPoints = it->second.toInt();
    }
    
    it = m.find(BC::Key::Overlay::GenericXY::xMin);
    if (it != m.end()) {
        d_xMin = it->second.toDouble();
    }
    
    it = m.find(BC::Key::Overlay::GenericXY::xMax);
    if (it != m.end()) {
        d_xMax = it->second.toDouble();
    }
    
    it = m.find(BC::Key::Overlay::GenericXY::yMin);
    if (it != m.end()) {
        d_yMin = it->second.toDouble();
    }
    
    it = m.find(BC::Key::Overlay::GenericXY::yMax);
    if (it != m.end()) {
        d_yMax = it->second.toDouble();
    }
    
    it = m.find(BC::Key::Overlay::GenericXY::filterMinX);
    if (it != m.end()) {
        d_filterMinX = it->second.toDouble();
    }
    
    it = m.find(BC::Key::Overlay::GenericXY::filterMaxX);
    if (it != m.end()) {
        d_filterMaxX = it->second.toDouble();
    }
}

#include "overlaytypes.h"

#include <data/storage/blackchirpcsv.h>
#include <data/experiment/experiment.h>
#include <QtMath>
#include <cmath>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QMutexLocker>


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
    QMutexLocker locker(&d_mutex);
    d_ft = ftData;

    // Restate the extrema under this overlay's ignore band so they agree with
    // yMax() regardless of the band the incoming Ft was processed under.
    d_ft.recomputeExtrema(d_autoScaleIgnoreMHz);
    setModified(true);
}

Ft BCExpOverlay::getFtData() const
{
    QMutexLocker locker(&d_mutex);
    return d_ft;
}

void BCExpOverlay::setAutoScaleIgnoreMHz(double mhz)
{
    QMutexLocker locker(&d_mutex);
    d_autoScaleIgnoreMHz = qMax(mhz,0.0);
    d_ft.recomputeExtrema(d_autoScaleIgnoreMHz);
    setModified(true);
}

double BCExpOverlay::getAutoScaleIgnoreMHz() const
{
    QMutexLocker locker(&d_mutex);
    return d_autoScaleIgnoreMHz;
}

double BCExpOverlay::yMax() const
{
    const auto lo = d_ft.loFreqMHz();
    double out = 0.0;

    for(int i=0; i<d_ft.size(); i++)
    {
        if(qAbs(d_ft.xAt(i) - lo) > d_autoScaleIgnoreMHz)
            out = qMax(out,qAbs(d_ft.at(i)));
    }

    // An ignore band wider than the data leaves nothing to scale against; the
    // full extent is a more useful answer than zero, which suppresses
    // autoscaling entirely.
    if(out <= 0.0)
        return OverlayBase::yMax();

    return out;
}

std::pair<double,double> BCExpOverlay::displayYRange() const
{
    if(d_autoScaleIgnoreMHz <= 0.0)
        return OverlayBase::displayYRange();

    // xyData() carries the X offset, so shift the band by the same amount to
    // compare it against the LO the FT was recorded with.
    const auto lo = d_ft.loFreqMHz() + getXOffset();
    const auto d = xyData();

    bool found = false;
    double yLo = 0.0, yHi = 0.0;

    for(const QPointF &p : d)
    {
        if(qAbs(p.x() - lo) <= d_autoScaleIgnoreMHz)
            continue;

        if(!found)
        {
            yLo = yHi = p.y();
            found = true;
        }
        else
        {
            yLo = qMin(yLo,p.y());
            yHi = qMax(yHi,p.y());
        }
    }

    if(!found)
        return OverlayBase::displayYRange();

    return {yLo,yHi};
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

    // The frequency axis and the ignore band arrive via _retrieveMetadata(),
    // which runs first, so the extrema can be recomputed here with the same
    // LO exclusion the FT was originally processed under. Publish under the
    // lock only; the file I/O above ran with no lock held.
    QMutexLocker locker(&d_mutex);
    d_ft.setData(ftData, 0.0, 0.0);
    d_ft.recomputeExtrema(d_autoScaleIgnoreMHz);
}

void BCExpOverlay::writeToDest()
{
    QString destFile = getDestFile();
    if(destFile.isEmpty())
        return;

    // Snapshot the FT data under the lock, then release it before doing
    // any file I/O -- this can run on a QtConcurrent worker
    // (OverlayStorage::addOverlay()) while the GUI thread calls
    // setFtData() on the same overlay.
    Ft ftSnapshot;
    {
        QMutexLocker locker(&d_mutex);
        ftSnapshot = d_ft;
    }

    QFile f(destFile);

    // Use BlackchirpCSV::writeY template function to write the Y data
    if(!BlackchirpCSV::writeY(f, ftSnapshot.yData(), QString("FT Magnitude")))
    {
        // Handle error if needed - file writing failed
        return;
    }
}

void BCExpOverlay::_storeMetadata(std::map<QString, QVariant, std::less<>> &m)
{
    using namespace BC::Key::Overlay;
    m.emplace(ftYMin, d_ft.yMin());
    m.emplace(ftYMax, d_ft.yMax());
    m.emplace(ftX0MHz, d_ft.xFirst());
    m.emplace(ftSpacingMHz, d_ft.xSpacing());
    m.emplace(ftLoFreqMHz, d_ft.loFreqMHz());
    m.emplace(ftShots, static_cast<qulonglong>(d_ft.shots()));
    m.emplace(ftAutoScaleIgnoreMHz, d_autoScaleIgnoreMHz);
}

void BCExpOverlay::_retrieveMetadata(const std::map<QString, QVariant, std::less<>> &m)
{
    using namespace BC::Key::Overlay;

    // ftYMin/ftYMax are written for the benefit of external readers but are not
    // consumed here: readFromDest() runs after this and recomputes them from
    // the magnitudes using the ignore band restored below.
    auto it = m.find(ftAutoScaleIgnoreMHz);
    if(it != m.end())
        d_autoScaleIgnoreMHz = qMax(it->second.toDouble(),0.0);

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
    // Default a freshly-created catalog overlay to a stick plot (a raw
    // line catalog is a stick spectrum). This is only the creation default:
    // retrieveMetadata() repopulates curve metadata from disk, so a user who
    // later switches to a line plot (e.g. after convolution) keeps that
    // choice across reloads.
    // QwtPlotCurve::CurveStyle ordinals: 0=Lines, 1=Sticks, 2=Steps, 3=Dots.
    // (The data layer cannot include qwt, so the literal is used directly.)
    setCurveMetadata(BC::Data::CurveKey::curveStyle, 1);
}

CatalogData CatalogOverlay::catalogData() const
{
    QMutexLocker locker(&d_mutex);
    return d_catalogData;
}

void CatalogOverlay::setCatalogData(const CatalogData &data)
{
    QMutexLocker locker(&d_mutex);
    if (d_catalogData != data) {
        d_catalogData = data;
        invalidateConvolutionCache();
        setModified(true);
    }
}

bool CatalogOverlay::convolutionEnabled() const
{
    QMutexLocker locker(&d_mutex);
    return d_convolutionEnabled;
}

void CatalogOverlay::setConvolutionEnabled(bool enabled)
{
    QMutexLocker locker(&d_mutex);
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
    QMutexLocker locker(&d_mutex);
    return d_lineshapeType;
}

void CatalogOverlay::setLineshapeType(LineshapeType type)
{
    QMutexLocker locker(&d_mutex);
    if (d_lineshapeType != type) {
        d_lineshapeType = type;
        invalidateConvolutionCache();
        setModified(true);
    }
}

double CatalogOverlay::linewidth() const
{
    QMutexLocker locker(&d_mutex);
    return d_linewidth;
}

void CatalogOverlay::setLinewidth(double width)
{
    QMutexLocker locker(&d_mutex);
    if (qAbs(d_linewidth - width) > 1e-6) {
        d_linewidth = width;
        invalidateConvolutionCache();
        setModified(true);
    }
}

double CatalogOverlay::convolutionMinFreq() const
{
    QMutexLocker locker(&d_mutex);
    return d_convolutionMinFreq;
}

double CatalogOverlay::convolutionMaxFreq() const
{
    QMutexLocker locker(&d_mutex);
    return d_convolutionMaxFreq;
}

void CatalogOverlay::setConvolutionFreqRange(double minFreq, double maxFreq)
{
    QMutexLocker locker(&d_mutex);
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
    QMutexLocker locker(&d_mutex);
    return d_numConvolutionPoints;
}

void CatalogOverlay::setNumConvolutionPoints(int numPoints)
{
    QMutexLocker locker(&d_mutex);
    if (d_numConvolutionPoints != numPoints) {
        d_numConvolutionPoints = numPoints;
        invalidateConvolutionCache();
        setModified(true);
    }
}

double CatalogOverlay::calculatePointSpacing() const
{
    QMutexLocker locker(&d_mutex);
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
    QMutexLocker locker(&d_mutex);
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
    QMutexLocker locker(&d_mutex);

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
    // Snapshot the catalog and convolution parameters under the lock, then
    // release it before doing any arithmetic. The convolution below is
    // O(numConvolutionPoints * transitions) and can run for a long time;
    // holding d_mutex across it would block every other access to this
    // overlay -- including GUI-thread plot reads -- for the duration. The
    // snapshot means a concurrent settings change (e.g. from a spinbox
    // edit while this runs on a worker thread) does not corrupt the
    // computation; it simply is not reflected in this particular result.
    CatalogData catalogSnapshot;
    LineshapeType lineshape;
    double linewidth, convMinFreq, convMaxFreq;
    int numPoints;
    {
        QMutexLocker locker(&d_mutex);
        if (d_catalogData.isEmpty())
            return QVector<QPointF>();

        catalogSnapshot = d_catalogData;
        lineshape = d_lineshapeType;
        linewidth = d_linewidth;
        convMinFreq = d_convolutionMinFreq;
        convMaxFreq = d_convolutionMaxFreq;
        numPoints = d_numConvolutionPoints;
    }

    // A non-positive point count has nothing to compute, and would
    // otherwise reach calculateChunkSize()/the chunked overload's
    // numChunks division with a zero-or-negative divisor. Not reachable
    // from the UI today (the spinbox floors at 100, and
    // ConvolutionOperation validates before calling in), but this is a
    // public method and must not crash on a degenerate input.
    if (numPoints <= 0)
        return QVector<QPointF>();

    // Everything below reads only local snapshots -- no lock held.

    //pre-filter transitions outside range; place into lightweight structures
    QVector<double> x0, y0;
    x0.reserve(catalogSnapshot.size());
    y0.reserve(catalogSnapshot.size());
    for(const auto &trans : catalogSnapshot.transitions())
    {
        if (trans.frequency >= convMinFreq && trans.frequency <= convMaxFreq)
        {
            x0.append(trans.frequency);
            y0.append(trans.intensity);
        }
    }

    // Generate frequency grid using number of points
    double pointSpacing = (numPoints <= 1) ? (convMaxFreq - convMinFreq)
                                            : (convMaxFreq - convMinFreq) / (numPoints - 1);
    QVector<QPointF> spectrum;
    spectrum.reserve(numPoints);

    //Store lineshape function pointer
    auto f = &CatalogOverlay::lorentzianProfile;
    if(lineshape == Gaussian)
        f = &CatalogOverlay::gaussianProfile;


    // Add contribution to each grid point
    for (int i = 0; i < numPoints; ++i) {
        double yy = 0.0;
        double gridFreq = convMinFreq + i * pointSpacing;
        for (int j = 0; (j < x0.size()) && (j < y0.size()); ++j) {
            yy += y0.at(j) * (this->*f)(gridFreq, x0.at(j), linewidth);
        }
        spectrum.append({gridFreq,yy});
    }

    return spectrum;
}

QVector<QPointF> CatalogOverlay::generateConvolvedSpectrum(ProgressCallback progressCallback) const
{
    // Same snapshot-then-unlock discipline as the no-callback overload
    // above: see the comment there for why the lock cannot span this
    // computation. The chunked loop below additionally invokes
    // progressCallback, which may call back into other overlay code
    // (e.g. checking cancellation) -- another reason the lock must not
    // be held here.
    CatalogData catalogSnapshot;
    LineshapeType lineshape;
    double linewidth, convMinFreq, convMaxFreq;
    int numPoints;
    {
        QMutexLocker locker(&d_mutex);
        if (d_catalogData.isEmpty())
            return QVector<QPointF>();

        catalogSnapshot = d_catalogData;
        lineshape = d_lineshapeType;
        linewidth = d_linewidth;
        convMinFreq = d_convolutionMinFreq;
        convMaxFreq = d_convolutionMaxFreq;
        numPoints = d_numConvolutionPoints;
    }

    // See the no-callback overload above: a non-positive point count
    // would otherwise reach calculateChunkSize()/the numChunks division
    // below with a zero-or-negative divisor.
    if (numPoints <= 0)
        return QVector<QPointF>();

    // Everything below reads only local snapshots -- no lock held.

    // Pre-filter transitions outside range; place into lightweight structures
    QVector<double> x0, y0;
    x0.reserve(catalogSnapshot.size());
    y0.reserve(catalogSnapshot.size());
    for(const auto &trans : catalogSnapshot.transitions())
    {
        if (trans.frequency >= convMinFreq && trans.frequency <= convMaxFreq)
        {
            x0.append(trans.frequency);
            y0.append(trans.intensity);
        }
    }

    // If no progress callback provided, fall back to the unchunked implementation.
    if (!progressCallback) {
        return generateConvolvedSpectrum();
    }

    // Calculate chunking parameters
    int chunkSize = calculateChunkSize(numPoints, x0.size());
    int numChunks = (numPoints + chunkSize - 1) / chunkSize;

    // Generate frequency grid using number of points
    double pointSpacing = (numPoints <= 1) ? (convMaxFreq - convMinFreq)
                                            : (convMaxFreq - convMinFreq) / (numPoints - 1);
    QVector<QPointF> spectrum;
    spectrum.reserve(numPoints);

    // Store lineshape function pointer
    auto f = &CatalogOverlay::lorentzianProfile;
    if(lineshape == Gaussian)
        f = &CatalogOverlay::gaussianProfile;

    // Process in chunks
    for (int chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
        // Calculate chunk boundaries
        int startIdx = chunkIdx * chunkSize;
        int endIdx = std::min(startIdx + chunkSize, numPoints);

        // Process chunk
        for (int i = startIdx; i < endIdx; ++i) {
            double yy = 0.0;
            double gridFreq = convMinFreq + i * pointSpacing;
            for (int j = 0; (j < x0.size()) && (j < y0.size()); ++j) {
                yy += y0.at(j) * (this->*f)(gridFreq, x0.at(j), linewidth);
            }
            spectrum.append({gridFreq, yy});
        }

        // Report progress and check for cancellation
        if (progressCallback) {
            int progressPercent = (chunkIdx + 1) * 100 / numChunks;
            QString message = QString("Processed %1/%2 chunks").arg(chunkIdx + 1).arg(numChunks);
            bool shouldContinue = progressCallback(progressPercent, message);
            if (!shouldContinue) {
                // Operation was cancelled - return empty result
                return QVector<QPointF>();
            }
        }
    }

    return spectrum;
}

int CatalogOverlay::calculateChunkSize(int numConvolutionPoints, int numTransitions) const
{
    // Target: 50M operations per chunk for ~100ms execution time
    const int targetOpsPerChunk = 50000000;
    const int opsPerPoint = numTransitions * 15; // Estimated ops per convolution point
    
    if (opsPerPoint <= 0) {
        return std::min(numConvolutionPoints, 10000); // Safe fallback
    }
    
    int idealChunkSize = targetOpsPerChunk / opsPerPoint;
    
    // Clamp to reasonable bounds: at least 1000 points, at most 100000 points
    int clampedSize = std::clamp(idealChunkSize, 1000, 100000);
    
    // Don't make chunks larger than the total number of points
    return std::min(clampedSize, numConvolutionPoints);
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
    QMutexLocker locker(&d_mutex);
    d_cacheState = CacheState::Invalid;
    // Invalidate base class cache to force refresh from _xyData()
    invalidateCache();
}

void CatalogOverlay::setCachePending()
{
    QMutexLocker locker(&d_mutex);
    d_cacheState = CacheState::Pending;
    // Invalidate base class cache to force refresh from _xyData()
    invalidateCache();
}

void CatalogOverlay::setCacheValid(const QVector<QPointF> &convolvedData)
{
    QMutexLocker locker(&d_mutex);
    d_convolvedCache = convolvedData;
    d_cacheState = CacheState::Valid;
    // Invalidate base class cache to force refresh from _xyData()
    invalidateCache();
}

bool CatalogOverlay::isCacheValid() const
{
    QMutexLocker locker(&d_mutex);
    return d_cacheState == CacheState::Valid;
}

bool CatalogOverlay::hasConvolvedData() const
{
    QMutexLocker locker(&d_mutex);
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
    if(destFile.isEmpty())
        return;

    // Snapshot the catalog data under the lock, then release it before
    // doing any serialization or file I/O -- the same discipline as
    // generateConvolvedSpectrum(). This can run on a QtConcurrent worker
    // (OverlayStorage::addOverlay()) while the GUI thread calls
    // setCatalogData() or a convolution updates the cache on the same
    // overlay.
    CatalogData catalogSnapshot;
    {
        QMutexLocker locker(&d_mutex);
        if(d_catalogData.isEmpty())
            return;
        catalogSnapshot = d_catalogData;
    }

    QFile f(destFile);

    // Prepare data vectors for BlackchirpCSV using QVariant for automatic formatting
    QVector<QVariant> frequencies, intensities, quantumNumbers, additionalData;

    frequencies.reserve(catalogSnapshot.size());
    intensities.reserve(catalogSnapshot.size());
    quantumNumbers.reserve(catalogSnapshot.size());
    additionalData.reserve(catalogSnapshot.size());

    for(int i = 0; i < catalogSnapshot.size(); ++i) {
        const TransitionData &trans = catalogSnapshot.at(i);
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

void CatalogOverlay::_storeMetadata(std::map<QString, QVariant, std::less<>> &m)
{
    using namespace BC::Key::Overlay::Catalog;
    
    m.emplace(sourceProgram, d_catalogData.sourceProgram());
    m.emplace(moleculeName, d_catalogData.moleculeName());
    m.emplace(BC::Key::Overlay::Catalog::convolutionEnabled, d_convolutionEnabled);
    m.emplace(BC::Key::Overlay::Catalog::lineshapeType, QVariant::fromValue(d_lineshapeType));
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

void CatalogOverlay::_retrieveMetadata(const std::map<QString, QVariant, std::less<>> &m)
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
        d_lineshapeType = BC::CSV::enumFromVariant<LineshapeType>(it->second,Lorentzian);
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
    
    // Invalidate cache after loading metadata
    invalidateConvolutionCache();
}

// GenericXYOverlay implementation

GenericXYOverlay::GenericXYOverlay() : OverlayBase(GenericXY)
{
}

QVector<QPointF> GenericXYOverlay::rawData() const
{
    QMutexLocker locker(&d_mutex);
    return d_rawData;
}

void GenericXYOverlay::setRawData(const QVector<QPointF> &data)
{
    QMutexLocker locker(&d_mutex);
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

double GenericXYOverlay::dataYMin() const
{
    return d_yMin;
}

double GenericXYOverlay::dataYMax() const
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
    if (destFile.isEmpty())
        return;

    // Snapshot the raw data under the lock, then release it before doing
    // any file I/O -- this can run on a QtConcurrent worker
    // (OverlayStorage::addOverlay()) while the GUI thread calls
    // setRawData() on the same overlay.
    QVector<QPointF> dataSnapshot;
    {
        QMutexLocker locker(&d_mutex);
        if (d_rawData.isEmpty())
            return;
        dataSnapshot = d_rawData;
    }

    QFile f(destFile);

    // Prepare data vectors for BlackchirpCSV
    QVector<QVariant> xData, yData;
    xData.reserve(dataSnapshot.size());
    yData.reserve(dataSnapshot.size());

    for (const QPointF &point : dataSnapshot) {
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

void GenericXYOverlay::_storeMetadata(std::map<QString, QVariant, std::less<>> &m)
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

void GenericXYOverlay::_retrieveMetadata(const std::map<QString, QVariant, std::less<>> &m)
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

#ifndef FT_H
#define FT_H

#include <QSharedDataPointer>

#include <QVector>
#include <QPointF>

class FtData;

/*!
 \brief Implicitly shared value type that stores a Fourier-transform magnitude spectrum.

 \c Ft holds a uniformly-spaced vector of magnitude values (produced by \c FtWorker::doFT),
 together with the frequency axis metadata needed to reconstruct absolute frequency coordinates:
 the starting frequency \c x0 (MHz), the uniform spacing (MHz/bin), and the local-oscillator
 frequency (MHz) used to orient the sideband. The class also caches the minimum and maximum
 magnitude values so that plots can autoscale without scanning the entire data vector.

 Like \c Fid, \c Ft uses Qt implicit sharing: copying is cheap and a deep copy occurs only
 when a mutating operation is called while the data is shared.

 \sa FtWorker, Fid
*/
class Ft
{
public:
    /*!
     \brief Default constructor; creates an empty spectrum with unit spacing and zero offset.
    */
    Ft();

    /*!
     \brief Constructs a spectrum with pre-allocated storage.
     \param numPnts Number of frequency bins to allocate (all initialized to zero).
     \param f0 Frequency of the first bin in MHz.
     \param spacing Frequency spacing between consecutive bins in MHz.
     \param loFreqMHz Local-oscillator frequency used to mark the DC bin for autoScale exclusion (MHz).
    */
    explicit Ft(int numPnts, double f0, double spacing, double loFreqMHz);

    /*!
     \brief Copy constructor (shallow; increments the shared reference count).
     \param rhs Spectrum to copy.
    */
    Ft(const Ft &rhs);

    /*!
     \brief Copy-assignment operator.
     \param rhs Spectrum to copy.
     \return Reference to this spectrum after assignment.
    */
    Ft &operator=(const Ft &rhs);

    /*!
     \brief Destructor.
    */
    ~Ft();

    /*!
     \brief Sets the magnitude at the specified bin and updates the cached min/max.

     If \a ignoreRange is greater than zero, bins whose frequency falls within
     \a ignoreRange MHz of the LO frequency are excluded from the min/max update.
     This prevents the DC spike from inflating the autoscale range.

     \param i Bin index.
     \param y Magnitude value.
     \param ignoreRange Frequency half-width (MHz) around the LO to exclude from autoscale tracking.
    */
    void setPoint(int i, double y, double ignoreRange = 0.0);

    /*!
     \brief Resizes the data vector and recomputes the cached min/max from the current contents.
     \param n New size in bins.
     \param ignoreRange Frequency half-width (MHz) around the LO to exclude from min/max tracking.
    */
    void resize(int n, double ignoreRange = 0.0);

    /*!
     \brief Returns a writable reference to the magnitude at bin \a i (no min/max update).
     \param i Bin index.
     \return Reference to the magnitude value.
    */
    double &operator[](int i);

    /*!
     \brief Reserves storage for at least \a n bins without changing the logical size.
     \param n Number of bins to reserve.
    */
    void reserve(int n);

    /*!
     \brief Releases excess capacity in the internal data vector.
    */
    void squeeze();

    /*!
     \brief Sets the local-oscillator reference frequency.
     \param f LO frequency in MHz.
    */
    void setLoFreq(double f);

    /*!
     \brief Sets the frequency of the first bin.
     \param d Starting frequency in MHz.
    */
    void setX0(double d);

    /*!
     \brief Sets the frequency spacing between consecutive bins.
     \param s Spacing in MHz per bin.
    */
    void setSpacing(double s);

    /*!
     \brief Appends a magnitude value to the end of the spectrum and updates the cached min/max.
     \param y Magnitude value to append.
    */
    void append(double y);

    /*!
     \brief Trims the spectrum to the frequency range [\a fmin, \a fmax].

     Bins outside the range are discarded and the starting frequency \c x0 is updated
     to the frequency of the first retained bin. The cached min/max is recomputed from
     the retained bins.

     \param fmin Minimum frequency to retain (MHz).
     \param fmax Maximum frequency to retain (MHz).
    */
    void trim(double fmin, double fmax);

    /*!
     \brief Sets the number of co-averaged shots represented by this spectrum.
     \param shots Shot count.
    */
    void setNumShots(quint64 shots);

    /*!
     \brief Replaces the data vector and sets the cached min/max explicitly.

     Intended for bulk-load scenarios where the caller has already computed the
     extrema, avoiding an O(n) scan.

     \param d New magnitude vector.
     \param yMin Pre-computed minimum magnitude.
     \param yMax Pre-computed maximum magnitude.
    */
    void setData(const QVector<double> d, double yMin, double yMax);

    /*!
     \brief Returns the number of frequency bins.
     \return Bin count.
    */
    int size() const;

    /*!
     \brief Returns \c true if the spectrum contains no bins.
     \return \c true when size() == 0.
    */
    bool isEmpty() const;

    /*!
     \brief Returns the magnitude at bin \a i (bounds-checked; aborts on out-of-range).
     \param i Bin index.
     \return Magnitude value.
    */
    double at(int i) const;

    /*!
     \brief Returns the magnitude at bin \a i, or 0.0 if out of range.
     \param i Bin index (may be out of range).
     \return Magnitude value, or 0.0.
    */
    double value(int i) const;

    /*!
     \brief Returns the magnitude of the first bin.
     \return Magnitude of bin 0.
    */
    double constFirst() const;

    /*!
     \brief Returns the magnitude of the last bin.
     \return Magnitude of bin size()-1.
    */
    double constLast() const;

    /*!
     \brief Returns the frequency of bin \a i.
     \param i Bin index.
     \return Frequency in MHz: x0 + i * spacing.
    */
    double xAt(int i) const;

    /*!
     \brief Returns the frequency of the first bin (same as x0).
     \return Frequency in MHz.
    */
    double xFirst() const;

    /*!
     \brief Returns the frequency of the last bin.
     \return Frequency in MHz: x0 + (size()-1) * spacing.
    */
    double xLast() const;

    /*!
     \brief Returns the [min, max] frequency range of the spectrum as a sorted pair.
     \return Pair where \c first <= \c second, both in MHz.
    */
    std::pair<double,double> xRange() const;

    /*!
     \brief Returns the frequency spacing between consecutive bins.
     \return Spacing in MHz per bin.
    */
    double xSpacing() const;

    /*!
     \brief Returns the lower edge of the spectrum's frequency range.
     \return Minimum of xFirst() and xLast() in MHz.
    */
    double minFreqMHz() const;

    /*!
     \brief Returns the upper edge of the spectrum's frequency range.
     \return Maximum of xFirst() and xLast() in MHz.
    */
    double maxFreqMHz() const;

    /*!
     \brief Returns the local-oscillator reference frequency.
     \return LO frequency in MHz.
    */
    double loFreqMHz() const;

    /*!
     \brief Returns the cached minimum magnitude value.

     The minimum is tracked incrementally by \c setPoint() and \c append() and
     recomputed by \c resize() and \c trim(). Bins near the LO may be excluded
     depending on the \a ignoreRange argument passed to \c setPoint().

     \return Minimum magnitude value.
    */
    double yMin() const;

    /*!
     \brief Returns the cached maximum magnitude value.

     Tracked and recomputed alongside \c yMin(). Use this for autoscaling plot axes.

     \return Maximum magnitude value.
    */
    double yMax() const;

    /*!
     \brief Builds a vector of frequency values, one per bin.
     \return Vector of frequencies in MHz, indexed from 0 to size()-1.
    */
    QVector<double> xData() const;

    /*!
     \brief Returns a copy of the raw magnitude data vector.
     \return Magnitude values indexed from 0 to size()-1.
    */
    QVector<double> yData() const;

    /*!
     \brief Builds a QVector of (frequency, magnitude) points suitable for plotting.
     \return XY data vector with one point per bin.
    */
    QVector<QPointF> toVector() const;

    /*!
     \brief Returns the number of co-averaged shots represented by this spectrum.
     \return Shot count set via \c setNumShots().
    */
    quint64 shots() const;

private:
    QSharedDataPointer<FtData> data; ///< Internal implicitly-shared data storage.
};

#endif // FT_H

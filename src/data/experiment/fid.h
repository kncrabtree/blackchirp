#ifndef FID_H
#define FID_H

#include <QSharedDataPointer>

#include <QVector>
#include <QPointF>
#include <QMetaType>
#include <QDataStream>

#include <data/experiment/rfconfig.h>

class FidData;

/*!
 \brief Implicitly shared value type that stores a single co-averaged free-induction decay (FID).

 Data are stored in an implicitly shared \c FidData pointer, which handles reference counting and
 copy-on-write semantics. Please see Qt documentation on implicit sharing for details.
 Copying an \c Fid is inexpensive because only the pointer is duplicated; a deep member-by-member
 copy occurs only when a mutating operation is called while more than one \c Fid shares the same
 underlying data.

 An \c Fid holds a vector of co-averaged integer samples (arbitrary digitizer units), a probe
 frequency (MHz), a sideband label, a per-sample time spacing (s), a voltage multiplier that
 converts the integer samples to volts, and a shot count. Non-const functions trigger a deep copy
 whenever the data is shared; const functions never do.

 The convenience alias \c FidList is a \c QVector<Fid> and carries the same implicit-sharing
 benefits at the list level.
*/
class Fid
{
public:
    /*!
     \brief Default constructor.

     Allocates a default FidData object with zero size, unit voltage multiplier,
     zero shots, and the upper sideband.
     */
    Fid();

    /*!
     \brief Copy constructor (shallow; increments the shared reference count).
     \param rhs FID to copy.
    */
    Fid(const Fid &rhs);

    /*!
     \brief Copy-assignment operator.
     \param rhs FID to copy.
     \return Reference to this FID after assignment.
    */
    Fid &operator=(const Fid &rhs);

    /*!
     \brief Destructor.

     Explicit declaration is required for the implicit-sharing mechanism. When the last
     reference to the underlying \c FidData is released the data is freed.
    */
    ~Fid();

    /*!
     \brief Sets the inter-sample time spacing (can trigger a deep copy).
     \param s New spacing in seconds.
    */
    void setSpacing(const double s);

    /*!
     \brief Sets the probe (local-oscillator) frequency (can trigger a deep copy).
     \param f New probe frequency in MHz.
    */
    void setProbeFreq(const double f);

    /*!
     \brief Replaces the raw sample vector (can trigger a deep copy).
     \param d New data vector of co-averaged integer samples.
    */
    void setData(const QVector<qint64> d);

    /*!
     \brief Sets the sideband label (can trigger a deep copy).
     \param sb Sideband; either \c RfConfig::UpperSideband or \c RfConfig::LowerSideband.
    */
    void setSideband(const RfConfig::Sideband sb);

    /*!
     \brief Sets the voltage multiplier used to convert integer samples to volts (can trigger a deep copy).
     \param vm Voltage multiplier in V/count.
    */
    void setVMult(const double vm);

    /*!
     \brief Sets the shot count (can trigger a deep copy).
     \param s Number of hardware shots co-averaged into the current data.
    */
    void setShots(const quint64 s);

    /*!
     \brief Detaches the internal data vector from any shared storage.

     Forces a deep copy of the \c QVector inside \c FidData, ensuring that subsequent
     raw-pointer operations do not alias another \c Fid's buffer.
    */
    void detach();

    /*!
     \brief Adds another FID's samples in-place, accumulating shot counts.
     \param other FID whose samples are added element-wise to this one.
     \return Reference to this FID after addition.
    */
    Fid &operator +=(const Fid other);

    /*!
     \brief Adds a raw integer vector in-place, incrementing the shot count by one.
     \param other Vector of integer samples whose length must match \c size().
     \return Reference to this FID after addition.
    */
    Fid &operator +=(const QVector<qint64> other);

    /*!
     \brief Adds a raw C-array in-place, incrementing the shot count by one.
     \param other Pointer to an array of at least \c size() integer samples.
     \return Reference to this FID after addition.
    */
    Fid &operator +=(const qint64 *other);

    /*!
     \brief Subtracts another FID's samples in-place, adjusting the shot count.

     If \a other has more shots than this FID, the subtraction is reversed so that
     the resulting shot count remains non-negative: the result represents
     \c other - \c this and carries \c other.shots() - \c this.shots() shots.

     \param other FID to subtract.
     \return Reference to this FID after subtraction.
    */
    Fid &operator -=(const Fid other);

    /*!
     \brief Adds another FID's samples with an optional time-domain shift.

     When \a shift is zero this is equivalent to \c operator+=. For non-zero
     \a shift, sample \c i of \a other is added to sample \c i+shift of this
     FID, with out-of-range indices treated as zero.

     \param other FID whose samples are added.
     \param shift Integer sample offset applied to \a other before accumulation.
    */
    void add(const Fid other, int shift);

    /*!
     \brief Copies a contiguous block from a raw C-array into the internal buffer.

     Uses \c memcpy to overwrite this FID's internal sample vector with \a size()
     samples read from \c other+offset. The shot count is incremented by one.

     \param other Pointer to the source sample array.
     \param offset Number of elements to skip at the start of \a other.
    */
    void copyAdd(const qint64 *other, const unsigned int offset = 0);

    /*!
     \brief Performs a rolling co-average against another FID.

     Accumulates \a other into this FID up to \a targetShots shots. Once the
     combined shot count would exceed \a targetShots, the accumulator is
     renormalized to exactly \a targetShots shots using integer rounding.

     \param other New FID to incorporate.
     \param targetShots Maximum number of shots to retain in the rolling average.
     \param shift Integer sample offset applied to \a other before accumulation.
    */
    void rollingAverage(const Fid other, quint64 targetShots, int shift = 0);

    /*!
     \brief Returns the number of samples in the data vector.
     \return Sample count.
    */
    int size() const;

    /*!
     \brief Returns \c true if the data vector contains no samples.
     \return \c true when size() == 0.
    */
    bool isEmpty() const;

    /*!
     \brief Returns the voltage at the specified index.

     The returned value is the normalized (per-shot) integer sample multiplied by
     the voltage multiplier: \c atNorm(i) * vMult().

     \param i Sample index.
     \return Voltage in volts.
    */
    double at(const int i) const;

    /*!
     \brief Returns the inter-sample time spacing.
     \return Spacing in seconds.
    */
    double spacing() const;

    /*!
     \brief Returns the probe (local-oscillator) frequency.
     \return Probe frequency in MHz.
    */
    double probeFreq() const;

    /*!
     \brief Builds a QVector of (time, voltage) points suitable for plotting.

     Each point has \c x = i * spacing() and \c y = at(i).

     \return XY data vector with one point per sample.
    */
    QVector<QPointF> toXY() const;

    /*!
     \brief Returns the data as a vector of voltage values.

     Equivalent to calling \c at(i) for each index; applies per-shot normalization
     and the voltage multiplier.

     \return Vector of voltage values in volts.
    */
    QVector<double> toVector() const;

    /*!
     \brief Returns the raw (un-normalized, un-scaled) co-averaged integer sample vector.
     \return Copy of the internal integer sample buffer.
    */
    QVector<qint64> rawData() const;

    /*!
     \brief Returns the raw integer sample at index \a i (bounds-checked via QVector::at).
     \param i Sample index.
     \return Raw integer sample value.
    */
    qint64 atRaw(const int i) const;

    /*!
     \brief Returns the raw integer sample at index \a i, or 0 if out of range.

     Uses \c QVector::value() so out-of-range accesses return 0 instead of
     asserting. Useful when adding FIDs with a time-domain shift.

     \param i Sample index (may be out of range).
     \return Raw integer sample value, or 0.
    */
    qint64 valueRaw(const int i) const;

    /*!
     \brief Returns the per-shot normalized sample at index \a i.

     When shots() > 1 the raw value is divided by the shot count; when shots() <= 1
     the raw value is returned as-is (cast to \c double).

     \param i Sample index.
     \return Normalized sample value (arbitrary units, no voltage scaling applied).
    */
    double atNorm(const int i) const;

    /*!
     \brief Returns the number of hardware shots co-averaged into the data.
     \return Shot count.
    */
    quint64 shots() const;

    /*!
     \brief Returns the sideband label.
     \return \c RfConfig::UpperSideband or \c RfConfig::LowerSideband.
    */
    RfConfig::Sideband sideband() const;

    /*!
     \brief Calculates the maximum spectral frequency that an FFT of this FID can represent.

     For the upper sideband the maximum frequency is
     probeFreq() + 1 / (2 * spacing() * 1e6) MHz. For the lower sideband the probe
     frequency is the maximum.

     \return Maximum frequency in MHz, or 0 if spacing() is zero.
    */
    double maxFreq() const;

    /*!
     \brief Calculates the minimum spectral frequency that an FFT of this FID can represent.

     For the lower sideband the minimum frequency is
     probeFreq() - 1 / (2 * spacing() * 1e6) MHz. For the upper sideband the probe
     frequency is the minimum.

     \return Minimum frequency in MHz, or 0 if spacing() is zero.
    */
    double minFreq() const;

    /*!
     \brief Returns the voltage multiplier.
     \return Voltage multiplier in V/count.
    */
    double vMult() const;

private:
    QSharedDataPointer<FidData> data; /*!< The internal implicitly-shared data storage object. */
};

/*!
 \brief Internal data for Fid (implicitly shared via QSharedData).

 Stores the inter-sample time spacing (s), the probe frequency (MHz), the voltage multiplier
 (V/count), the shot count, the sideband label, and the raw co-averaged integer sample vector.
 This class is an implementation detail of \c Fid and is not part of the public API.
*/
class FidData : public QSharedData {
public:
    FidData () {}
    double spacing{1.0};   ///< Inter-sample time spacing in seconds.
    double probeFreq{0.0}; ///< Local-oscillator probe frequency in MHz.
    double vMult{1.0};     ///< Voltage multiplier (V/count) applied when converting to volts.
    quint64 shots{0};      ///< Number of hardware shots accumulated in \c fid.
    QVector<qint64> fid;   ///< Co-averaged raw integer sample buffer.
    RfConfig::Sideband sideband{RfConfig::UpperSideband}; ///< Sideband label for frequency-axis orientation.
};

Q_DECLARE_METATYPE(Fid)
Q_DECLARE_TYPEINFO(Fid,Q_MOVABLE_TYPE);
using FidList = QVector<Fid>; ///< Convenience alias for a list of \c Fid objects.
Q_DECLARE_TYPEINFO(FidList,Q_MOVABLE_TYPE);

#endif // FID_H

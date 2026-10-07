#ifndef FCUTUNESWEEP_H
#define FCUTUNESWEEP_H

#include <vector>

#include <QObject>
#include <QString>

/*!
 * \file fcutunesweep.h
 * \brief \c FcuTuneSweep — a trim sweep that locates the phase-match peak of
 *        a frequency-conversion stage from the reference-channel signal.
 */
namespace BC::FcuTune {
Q_NAMESPACE

/*!
 * \brief Outcome of a completed sweep's peak fit.
 */
enum class Status {
    Success,          ///< Peak located inside the window; \c Result::center is usable.
    PeakAtEdge,       ///< Maximum at the first or last point; the true peak may lie outside the window.
    LowContrast,      ///< Signal varies too little across the window to locate a peak.
    Clipped,          ///< At least one reference waveform saturated the digitizer.
    InsufficientData, ///< Too few points, or a point with no accepted waveforms.
    FitFailed,        ///< The peak model could not be fit (e.g. no curvature).
    MoveFailed,       ///< The stage could not be moved to a sweep point.
    Aborted           ///< The sweep was stopped before completion.
};
Q_ENUM_NS(Status)

/*!
 * \brief Sweep parameters. Trims are in the stage's native units (e.g. motor steps).
 */
struct Settings {
    double halfWidth{2000.0};   ///< Window extends this far either side of the starting trim.
    int points{11};             ///< Number of trim values visited (at least 3).
    int waveformsPerPoint{20};  ///< Accepted waveforms averaged at each trim value.
    int discardPerPoint{2};     ///< Waveforms ignored after each move before accepting any.
    double minContrast{0.2};    ///< Minimum (max - min)/max across the window for a valid peak.
    int direction{1};           ///< +1 visits trims in increasing order, -1 in decreasing order.
};

/*!
 * \brief Sweep outcome. The per-point vectors are in visiting order.
 */
struct Result {
    // Default-constructible and copyable so it can travel in queued signals.
    Status status{Status::InsufficientData};
    double center{0.0};             ///< Fitted peak trim (valid when \c status is Success).
    double centerUncertainty{0.0};  ///< One-sigma uncertainty of \c center, from the fit covariance.
    double contrast{0.0};           ///< (max - min)/max of the oriented point means.
    QString message;                ///< Human-readable explanation of \c status.
    std::vector<double> trims;      ///< Trim value at each point.
    std::vector<double> means;      ///< Mean reference integral at each point.
    std::vector<double> stdErrs;    ///< Standard error of each mean.

    bool success() const { return status == Status::Success; }
};

}

/*!
 * \brief Hardware-free state machine and peak fit for one trim sweep.
 *
 * Visits \c Settings::points evenly spaced trim values spanning
 * \c [start - halfWidth, start + halfWidth] in the order set by
 * \c Settings::direction (a stage's preferred trim direction, so successive
 * moves avoid backlash pre-moves). At each point, the caller moves the
 * stage to currentTrim(), then feeds one reference integral per waveform to
 * addWaveform(); the first \c discardPerPoint are ignored, so waveforms in
 * flight across the move do not contaminate the point. When addWaveform()
 * reports the point complete, the caller calls advance() and moves to the
 * next trim, until isComplete().
 *
 * result() fits the point means. The reference signal is oriented so the
 * peak is a maximum (a negative-going photodiode pulse integrates to a
 * negative value), then a Gaussian is fit to the contiguous run of points
 * at or above half the maximum (a weighted parabola in the logarithm of
 * the signal), which approximates the sinc² phase-match curve near its
 * peak. Validation rejects a maximum at the window edge, a contrast below
 * \c minContrast, any clipped waveform, and a fit without negative
 * curvature or with a vertex outside the fitted run.
 */
class FcuTuneSweep
{
public:
    /*!
     * \brief Construct a sweep centered on \a startTrim. \a settings is
     *        sanitized: at least 3 points, nonnegative counts, at least one
     *        waveform per point, positive half-width, direction ±1.
     */
    FcuTuneSweep(const BC::FcuTune::Settings &settings, double startTrim);

    const BC::FcuTune::Settings &settings() const { return d_settings; }
    double startTrim() const { return d_startTrim; }
    const std::vector<double> &trims() const { return d_trims; }

    int currentIndex() const { return d_index; }
    //! Trim of the point currently being acquired (the last point once complete).
    double currentTrim() const;
    bool isComplete() const { return d_index >= static_cast<int>(d_trims.size()); }

    /*!
     * \brief Offer one waveform's reference integral to the current point.
     *
     * \a clipped flags a saturated reference waveform; it is counted toward
     * the point and makes the final result Clipped.
     *
     * \return \c true when the current point has its full count of
     * accepted waveforms.
     */
    bool addWaveform(double refIntegral, bool clipped = false);

    //! Move to the next point. \return \c true if a further point remains.
    bool advance();

    //! Progress across all points, 0–1000.
    int perMilComplete() const;

    //! Mean and standard error of the accepted waveforms at point \a i (NaN if none).
    std::pair<double,double> pointStats(int i) const;

    /*!
     * \brief Fit the completed (or partial) sweep. Points with no accepted
     *        waveforms make the result InsufficientData.
     */
    BC::FcuTune::Result result() const;

    /*!
     * \brief Fit \a means (with standard errors \a stdErrs) at \a trims.
     *
     * The validation and fit described in the class comment, exposed for
     * direct testing. \a trims need not be sorted.
     */
    static BC::FcuTune::Result fitPeak(const std::vector<double> &trims,
                                        const std::vector<double> &means,
                                        const std::vector<double> &stdErrs,
                                        double minContrast);

private:
    struct PointData {
        int seen{0};
        int accepted{0};
        double sum{0.0};
        double sumSq{0.0};
    };

    BC::FcuTune::Settings d_settings;
    double d_startTrim;
    std::vector<double> d_trims;
    std::vector<PointData> d_points;
    int d_index{0};
    bool d_clipped{false};
};

#endif // FCUTUNESWEEP_H

#ifndef FTWORKER_H
#define FTWORKER_H

#include <QObject>
#include <QVector>
#include <QPointF>
#include <QPair>
#include <QTimer>
#include <memory>

#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

#include <data/analysis/analysis.h>
#include <data/analysis/ft.h>
#include <data/experiment/fid.h>

class QReadWriteLock;

/*!
 \brief Processes \c Fid objects into magnitude spectra using the GNU Scientific Library FFT.

 \c FtWorker computes mixed-radix Fast Fourier Transforms via the GSL real-data FFT routines,
 which are most efficient when the FID length can be factored into powers of 2, 3, and 5.
 In addition to the FFT itself, the worker applies an optional pipeline of time-domain
 preprocessing steps: truncation to a sub-window (\c FidProcessingSettings::startUs /
 \c endUs), DC removal, exponential apodization (\c expFilter), zero-padding
 (\c zeroPadFactor), and one of several \c FtWindowFunction apodization windows
 (Bartlett, Blackman, Blackman-Harris, Hamming, Hanning, Kaiser-Bessel).

 The magnitude scale of the output \c Ft is controlled by \c FidProcessingSettings::units:
 the enumeration values map directly to powers of ten (e.g. \c FtmV = 3 → multiply by
 \c 10³ / rawSize).

 \section ftworker-threading Threading model

 \c FtWorker is a \c QObject but is \e not moved to a dedicated \c QThread. Instead,
 callers invoke \c doFT(), \c doFtDiff(), and \c processSideband() through
 \c QtConcurrent::run, which schedules each call on the global thread pool. The signals
 \c ftDone, \c fidDone, \c ftDiffDone, and \c sidebandDone are emitted from the
 thread-pool thread. Connecting them to slots in the UI thread requires queued connections
 (the default when objects live on different threads).

 When called from a non-UI thread (e.g. a \c BatchManager), the return values of
 \c doFT() and \c filterFid() can be used directly without waiting for the signals.
 The \a id parameter of \c doFT() controls whether signals are emitted: pass \c id >= 0
 to get signal delivery (asynchronous path) or \c id = -1 to suppress signals
 (synchronous path).

 GSL workspace allocation is guarded by \c pu_fftLock and spline allocation by
 \c pu_splineLock, so the same \c FtWorker instance can be called concurrently from
 multiple thread-pool threads provided each concurrent call uses a different code path.
 The window-function cache is guarded by \c pu_winfLock.

 \section ftworker-resources Resource management

 GSL wavetables, workspaces, and spline objects are allocated lazily on first use and
 freed by the destructor. When idle-cleanup is enabled via \c setIdleCleanupEnabled(),
 resources are also freed automatically after a 5-minute inactivity timeout
 (\c cleanupResources()). Call \c resetIdleTimer() after each processing operation
 to restart the countdown.

 \sa Fid, Ft
*/
class FtWorker : public QObject
{
    Q_OBJECT
    friend class FtWorkerTest;
public:

    /*!
     \brief Output magnitude units for the Fourier-transform result.

     The enumeration value is the base-10 exponent applied to the raw GSL magnitude
     before dividing by the FID length. For example, \c FtmV = 3 means the output
     is in millivolts.
    */
    enum FtUnits {
        FtV  = 0, ///< Volts.
        FtmV = 3, ///< Millivolts.
        FtuV = 6, ///< Microvolts.
        FtnV = 9  ///< Nanovolts.
    };
    Q_ENUM(FtUnits)

    /*!
     \brief Window functions applied to the FID before the FFT.

     Windowing reduces spectral leakage at the cost of reduced frequency resolution.
     \c None applies a rectangular (uniform) window.
    */
    enum FtWindowFunction {
        None,           ///< No windowing (rectangular window).
        Bartlett,       ///< Bartlett (triangular) window.
        Blackman,       ///< Blackman window.
        BlackmanHarris, ///< Blackman-Harris window.
        Hamming,        ///< Hamming window.
        Hanning,        ///< Hanning (Hann) window.
        KaiserBessel    ///< Kaiser-Bessel window with \f$\beta = 14\f$.
    };
    Q_ENUM(FtWindowFunction)

    /*!
     \brief Method used to combine sideband FT magnitudes during LO-scan co-averaging.
    */
    enum DeconvolutionMethod {
        Harmonic_Mean, ///< Combine magnitudes using the shot-weighted harmonic mean.
        Geometric_Mean ///< Combine magnitudes using the shot-weighted geometric mean.
    };
    Q_ENUM(DeconvolutionMethod)

    /*!
     \brief Parameters that control FID preprocessing and FFT output.

     All fields must be set before passing an instance to \c doFT() or \c filterFid().
    */
    struct FidProcessingSettings {
        double startUs;             ///< Start of the time window to process, in microseconds (0 = beginning of FID).
        double endUs;               ///< End of the time window to process, in microseconds (0 = end of FID).
        double expFilter;           ///< Exponential apodization time constant in microseconds (0 = disabled).
        int zeroPadFactor;          ///< Zero-padding multiplier: 0 = none, 1 = next power of 2 × 2, 2 = × 4.
        bool removeDC;              ///< When \c true, subtract the mean of the windowed FID before the FFT.
        FtUnits units;              ///< Output magnitude units; see \c FtUnits.
        double autoScaleIgnoreMHz;  ///< Frequency half-width around the LO (MHz) to exclude from autoscale min/max tracking.
        FtWindowFunction windowFunction; ///< Apodization window applied after truncation and DC removal.
    };

    /*!
     \brief Return value of \c filterFid().
    */
    struct FilterResult {
        QVector<double> fid; ///< Preprocessed (floating-point) FID samples ready for FFT input.
        double min{0.0};     ///< Minimum sample value after all preprocessing steps.
        double max{0.0};     ///< Maximum sample value after all preprocessing steps.
    };

    /*!
     \brief Input bundle for a single step of an LO-scan sideband co-average.

     \c processSideband() is called once per LO frequency step. When \c currentIndex
     reaches \c totalFids - 1, the accumulated result is emitted via \c sidebandDone().
    */
    struct SidebandProcessingData {
        FidList fl;                                   ///< FID list for this LO step.
        int frame{0};                                 ///< Frame index within \c fl to process (-1 = average all).
        int totalFids{0};                             ///< Total number of LO-frequency steps in the full scan.
        int currentIndex{0};                          ///< Zero-based index of this step within the scan.
        double minOffset{-1.0};                       ///< Minimum frequency offset from the LO to include (MHz).
        double maxOffset{-1.0};                       ///< Maximum frequency offset from the LO to include (MHz).
        std::pair<double,double> loRange{0.0,0.0};    ///< [min, max] LO probe-frequency range of the full scan (MHz).
        RfConfig::Sideband sideband{RfConfig::UpperSideband}; ///< Sideband to process when not in double-sideband mode.
        bool doubleSideband{false};                   ///< When \c true, process both sidebands simultaneously.
        DeconvolutionMethod dcMethod{Harmonic_Mean};  ///< Method for combining overlapping FT bins across LO steps.
    };

    /*!
     \brief Constructor; initializes GSL pointer members to null and creates internal locks and idle timer.
     \param parent Optional parent object.
    */
    explicit FtWorker(QObject *parent = nullptr);

    /*!
     \brief Destructor; frees all GSL wavetables, workspaces, and spline objects.
    */
    ~FtWorker();

signals:
    /*!
     \brief Emitted when a \c doFT() call completes.

     Only emitted when \c doFT() is called with \a id >= 0. The \a id value is
     forwarded from the \c doFT() call so the receiver can match results to requests.

     \param ft The completed magnitude spectrum.
     \param id Caller-supplied identifier passed to \c doFT().
    */
    void ftDone(Ft ft, int id);

    /*!
     \brief Emitted after FID filtering is complete, carrying the preprocessed FID for display.

     Only emitted when \c doFT() is called with \a id >= 0.

     \param fid Preprocessed floating-point FID samples (after windowing/truncation/DC removal).
     \param spacing Inter-sample spacing in microseconds.
     \param min Minimum sample value after preprocessing.
     \param max Maximum sample value after preprocessing.
     \param shots Number of co-averaged shots.
     \param id Caller-supplied identifier passed to \c doFT().
    */
    void fidDone(QVector<double> fid, double spacing, double min, double max, quint64 shots, int id);

    /*!
     \brief Emitted when a \c doFtDiff() call completes.
     \param ft Difference spectrum (reference FT minus comparison FT).
    */
    void ftDiffDone(Ft ft);

    /*!
     \brief Emitted when the final step of \c processSideband() completes.
     \param ft The fully co-averaged LO-scan sideband spectrum.
    */
    void sidebandDone(Ft ft);

public slots:
    /*!
     \brief Filters and Fourier-transforms a list of FIDs.

     Selects the FID at \a frame (or averages all FIDs when \a frame < 0), applies
     the preprocessing pipeline described by \a settings, then computes the
     mixed-radix GSL FFT and converts coefficients to a magnitude spectrum.

     The magnitude scale factor is \c 10^(settings.units) / rawFidSize. The DC bin
     at the probe frequency is zeroed regardless of settings.

     When \a id >= 0, \c fidDone() and \c ftDone() signals are emitted (useful for
     asynchronous callers in the UI thread). When \a id == -1 the signals are
     suppressed and only the return value is used (useful for synchronous callers
     such as \c BatchManager).

     GSL workspace and wavetable are allocated or reallocated when the FID length
     changes; allocation is protected by \c pu_fftLock.

     \param fl List of FIDs to process.
     \param settings Preprocessing and FFT parameters.
     \param frame Frame index to process; negative values average all frames.
     \param id Caller-supplied identifier forwarded to \c ftDone() and \c fidDone(); -1 suppresses signals.
     \param doubleSideband When \c true, produce a symmetric double-sideband spectrum.
     \return The completed magnitude spectrum, or a default-constructed \c Ft if \a fl is empty.
    */
    Ft doFT(const FidList fl, const FtWorker::FidProcessingSettings &settings, int frame = 0, int id = -1, bool doubleSideband = false);

    /*!
     \brief Computes the difference spectrum between two FID lists and emits \c ftDiffDone().

     Each FID in \a refList and \a diffList must have matching length, sideband, and spacing.
     When the two spectra share the same LO frequency, the difference is computed
     bin-by-bin. When the LO frequencies differ, the \a diffList spectrum is resampled
     onto the \a refList frequency grid using cubic spline interpolation before subtraction.

     \param refList Reference FID list.
     \param diffList Comparison FID list.
     \param refFrame Frame index for the reference.
     \param diffFrame Frame index for the comparison.
     \param settings Preprocessing and FFT parameters applied to both lists.
    */
    void doFtDiff(const FidList refList, const FidList diffList, int refFrame, int diffFrame, const FtWorker::FidProcessingSettings &settings);

    /*!
     \brief Processes one step of an LO-scan sideband co-average and emits \c sidebandDone() on the last step.

     Call this slot once per LO frequency step with monotonically increasing
     \c SidebandProcessingData::currentIndex values from 0 to
     \c SidebandProcessingData::totalFids - 1. The internal \c LoScanData accumulator
     is reset when \c currentIndex == 0. On the final step, the accumulated spectrum
     is emitted via \c sidebandDone().

     Individual FT bins are combined using the method specified by
     \c SidebandProcessingData::dcMethod (harmonic or geometric mean, weighted by shot
     count). Fractional grid alignment between LO steps is resolved by linear
     interpolation of the nearest two bins.

     \param d Processing parameters and FID data for this LO step.
     \param settings Preprocessing and FFT parameters.
    */
    void processSideband(const SidebandProcessingData &d, const FidProcessingSettings &settings);

    /*!
     \brief Applies the preprocessing pipeline to a single FID without computing an FFT.

     Performs truncation, DC removal, exponential apodization, window function application,
     and zero-padding as specified by \a settings. This is called internally by \c doFT()
     and can also be used independently to obtain a filtered FID for display.

     \param fid Source FID.
     \param settings Preprocessing parameters.
     \return \c FilterResult containing the preprocessed floating-point samples and their min/max.
    */
    FilterResult filterFid(const Fid fid, const FtWorker::FidProcessingSettings &settings);

    /*!
     \brief Frees all GSL wavetables, workspaces, spline objects, and window-function cache.

     Safe to call from any thread; each resource group is protected by its own
     read/write lock. Also stops the idle timer. After this call, the next
     \c doFT() invocation will reallocate all necessary GSL resources.
    */
    void cleanupResources();

    /*!
     \brief Restarts the idle-cleanup countdown timer.

     Has no effect when idle cleanup is disabled or no resources are currently
     allocated. The timer is restarted via a queued \c QMetaObject::invokeMethod
     call to ensure it runs on the correct thread.
    */
    void resetIdleTimer();

    /*!
     \brief Enables or disables automatic resource cleanup on idle timeout.

     When \a enabled is \c false, the idle timer is stopped immediately. When
     \a enabled is \c true, the timer will fire after 5 minutes of inactivity and
     call \c cleanupResources().

     \param enabled \c true to enable idle cleanup, \c false to disable.
    */
    void setIdleCleanupEnabled(bool enabled);

private:
    std::unique_ptr<QReadWriteLock> pu_fftLock;    ///< Guards GSL FFT wavetable and workspace allocation.
    std::unique_ptr<QReadWriteLock> pu_splineLock; ///< Guards GSL spline and accelerator allocation.
    std::unique_ptr<QReadWriteLock> pu_winfLock;   ///< Guards the cached window-function vector.
    gsl_fft_real_wavetable *real; ///< Wavetable for GNU Scientific Library FFT operations.
    gsl_fft_real_workspace *work; ///< Workspace for GNU Scientific Library FFT operations.
    int d_numPnts; ///< FID length for which the current wavetable and workspace were allocated.

    gsl_spline *p_spline;         ///< GSL cubic spline object used for cross-LO FT resampling.
    gsl_interp_accel *p_accel;    ///< GSL interpolation accelerator associated with \c p_spline.
    int d_numSplinePoints;        ///< Number of points for which \c p_spline was allocated.

    FtWindowFunction d_lastWinf{None}; ///< Window function used to compute the cached \c d_winf vector.
    int d_lastWinSize{0};              ///< Size for which \c d_winf was last computed.

    /*!
     \brief Per-instance accumulator for an in-progress LO-scan sideband co-average.

     Populated during successive \c processSideband() calls; reset when
     \c SidebandProcessingData::currentIndex == 0.
    */
    struct LoScanData {
        QVector<double> ftData;              ///< Accumulated magnitude values on the output frequency grid.
        uint ftPoints{0};                    ///< Number of output frequency grid points.
        double ftSpacing{0.0};               ///< Output frequency grid spacing in MHz.
        quint64 totalShots{0};               ///< Total co-averaged shots accumulated so far.
        std::pair<double,double> ftXRange{0.0,0.0}; ///< [start, end] frequency range of the output grid (MHz).
        QVector<quint64> counts;             ///< Per-bin shot-count accumulator used for weighted averaging.

        /// Returns the output grid index nearest to frequency \a f.
        uint indexOf(const double f) {
            return f < ftXRange.first ? 0 : static_cast<int>(floor((f-ftXRange.first)/ftSpacing));
        }

        /// Returns the frequency corresponding to output grid index \a index.
        double frequency(int index) {
            return ftXRange.first + index*ftSpacing;
        }

        /// Returns the fractional grid offset of \a f relative to its nearest grid point (in units of ftSpacing).
        double relDistance(double f) {
            auto fNearest = frequency(indexOf(f));
            return qAbs(f-fNearest)/ftSpacing;
        }

    } d_loScanData;

    /*!
     \brief Resamples the frequency axis of \a ft onto a grid starting at \a f0 with spacing \a spacing.

     Uses a GSL cubic spline for interpolation. Points outside the valid range of
     \a ft are set to zero. Spline allocation is guarded by \c pu_splineLock.

     \param f0 Starting frequency of the target grid (MHz).
     \param spacing Target frequency spacing (MHz).
     \param ft Source spectrum to resample.
     \return Pair of (resampled magnitude vector, actual starting frequency of the output grid in MHz).
    */
    QPair<QVector<double>, double> resample(double f0, double spacing, const Ft ft);

    //store a precalculated window function for speed
    QVector<double> d_winf; ///< Cached window-function coefficients; recomputed when size or type changes.
    void makeWinf(int n, FtWindowFunction f);      ///< Recomputes \c d_winf for length \a n and function \a f.
    void winBartlett(int n);                        ///< Fills \c d_winf with Bartlett coefficients.
    void winBlackman(int n);                        ///< Fills \c d_winf with Blackman coefficients.
    void winBlackmanHarris(int n);                  ///< Fills \c d_winf with Blackman-Harris coefficients.
    void winHamming(int n);                         ///< Fills \c d_winf with Hamming coefficients.
    void winHanning(int n);                         ///< Fills \c d_winf with Hanning coefficients.
    void winKaiserBessel(int n, double beta);       ///< Fills \c d_winf with Kaiser-Bessel coefficients for parameter \a beta.

    void clearSplineMemory(); ///< Frees the GSL spline and accelerator objects under \c pu_splineLock.

private slots:
    void onIdleTimeout();  ///< Triggered by the idle timer; calls \c cleanupResources() when idle cleanup is enabled.
    void startIdleTimer(); ///< Starts or restarts the idle timer; must be called from the object's thread.

private:
    std::unique_ptr<QTimer> pu_idleTimer;   ///< Single-shot timer that fires after 5 minutes of inactivity.
    bool d_resourcesAllocated{false};       ///< \c true when any GSL resource has been allocated since the last cleanup.
    bool d_idleCleanupEnabled{false};       ///< \c true when the idle-cleanup mechanism is active.

};

Q_DECLARE_METATYPE(FtWorker::FidProcessingSettings)

#endif // FTWORKER_H

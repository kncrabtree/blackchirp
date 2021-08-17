#ifndef FTWORKER_H
#define FTWORKER_H

#include <QObject>
#include <QVector>
#include <QPointF>
#include <QPair>

#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

#include <data/analysis/analysis.h>
#include <data/analysis/ft.h>
#include <data/experiment/fid.h>



/*!
 \brief Class that handles processing of FIDs

 This class uses algorithms from the GNU Scientific library to perform Fast Fourier Transforms on Fid objects.
 The FFT algorithms are based on a mixed-radix approach that is particularly efficient when the Fid length can be factored many times into multiples of 2, 3, and 5.
 Further details about the algorithm can be found at http://www.gnu.org/software/gsl/manual/html_node/Mixed_002dradix-FFT-routines-for-real-data.html.
 In addition, an Fid can be processed by high pass filtering, exponential filtering, and truncation.

 The FtWorker is designed to operate in its own thread of execution.
 Communication can proceed either through signals and slots, or by direct function calls if the calling thread is not the UI thread (e.g., in a BatchManager object).
 Two slots are available, doFt() and filterFid(), and each has a corresponding signal ftDone() and fidDone() that are emitted when the operation is complete.
 Both slots also return the values that are emitted for use in direct function calls; doFt() calls filterFid() internally, for example.

*/
class FtWorker : public QObject
{
    Q_OBJECT
public:
    enum FtUnits {
        FtV = 0,
        FtmV = 3,
        FtuV = 6,
        FtnV = 9
    };
    Q_ENUM(FtUnits)

    enum FtWindowFunction {
        None,
        Bartlett,
        Blackman,
        BlackmanHarris,
        Hamming,
        Hanning,
        KaiserBessel
    };
    Q_ENUM(FtWindowFunction)

    struct FidProcessingSettings {
        double startUs;
        double endUs;
        int zeroPadFactor;
        bool removeDC;
        FtUnits units;
        double autoScaleIgnoreMHz;
        FtWindowFunction windowFunction;
    };

    /*!
     \brief Constructor. Does nothing

     \param parent
    */
    explicit FtWorker(int i, QObject *parent = nullptr);
    ~FtWorker();

    const int d_id;

signals:
    /*!
     \brief Emitted when FFT is complete
    */
    void ftDone(const Ft ft, int);
    /*!
     \brief Emitted when Fid filtering is complete

     \param fid The filtered Fid
    */
    void fidDone(const QVector<QPointF> fid, int);

    void ftDiffDone(const Ft ft);

public slots:
    /*!
     \brief Filters and performs FFT operation on an Fid

     \param fid Fid to analyze
     \return QPair<QVector<QPointF>, double> Resulting FT magnitude spectrum in XY format and maximum Y value
    */
    Ft doFT(const Fid fid, const FtWorker::FidProcessingSettings &settings);
    void doFtDiff(const Fid ref, const Fid diff, const FtWorker::FidProcessingSettings &settings);
    Ft processSideband(const FidList fl, const FtWorker::FidProcessingSettings &settings, RfConfig::Sideband sb, double minFreq = 0.0, double maxFreq = -1.0);
    void processBothSidebands(const FidList fl, const FtWorker::FidProcessingSettings &settings, double minFreq = 0.0, double maxFreq = -1.0);

    /*!
     \brief Perform truncation, high-pass, and exponential filtering on an Fid

     \param f Fid to filter
     \return QVector<double> Filtered Fid
    */
    QVector<double> filterFid(const Fid fid, const FtWorker::FidProcessingSettings &settings);

    void prepareForDisplay(const QVector<double> fid, double spacing);

private:
    gsl_fft_real_wavetable *real; /*!< Wavetable for GNU Scientific Library FFT operations */
    gsl_fft_real_workspace *work; /*!< Memory for GNU Scientific Library FFT operations */
    int d_numPnts; /*!< Number of points used to allocate last wavetable and workspace */

    gsl_spline *p_spline;
    gsl_interp_accel *p_accel;
    int d_numSplinePoints;

    FidProcessingSettings d_lastProcSettings;

    QList<Ft> makeSidebandList(const FidList fl, const FtWorker::FidProcessingSettings &settings, RfConfig::Sideband sb, double minFreq = 0.0, double maxFreq = -1.0);
    Ft resample(double f0, double spacing, const Ft ft);


    //store a precalculated window function for speed
    QVector<double> d_winf;
    void makeWinf(int n,FtWindowFunction f);
    void winBartlett(int n);
    void winBlackman(int n);
    void winBlackmanHarris(int n);
    void winHamming(int n);
    void winHanning(int n);
    void winKaiserBessel(int n, double beta);

};

Q_DECLARE_METATYPE(FtWorker::FidProcessingSettings)

#endif // FTWORKER_H

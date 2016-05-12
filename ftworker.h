#ifndef FTWORKER_H
#define FTWORKER_H

#include <QObject>
#include <QVector>
#include <QPointF>
#include <QPair>

#include <gsl/gsl_fft_real.h>

#include "fid.h"

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

TODO: Eliminate artifacts from QtFTM; add in better processing options (zero padding; possibility of windowing, etc)

*/
class FtWorker : public QObject
{
    Q_OBJECT
public:
    /*!
     \brief Constructor. Does nothing

     \param parent
    */
    explicit FtWorker(QObject *parent = 0);

signals:
    /*!
     \brief Emitted when FFT is complete

     \param ft FT data in XY format
     \param max Maximum Y value of FT
    */
    void ftDone(QVector<QPointF> ft, double max);
    /*!
     \brief Emitted when Fid filtering is complete

     \param fid The filtered Fid
    */
    void fidDone(QVector<QPointF> fid);

    void ftDiffDone(QVector<QPointF> ft, double min, double max);

public slots:
    /*!
     \brief Filters and performs FFT operation on an Fid

     \param fid Fid to analyze
     \return QPair<QVector<QPointF>, double> Resulting FT magnitude spectrum in XY format and maximum Y value
    */
    QPair<QVector<QPointF>,double> doFT(const Fid fid);
    void doFtDiff(const Fid ref, const Fid diff);

    /*!
     \brief Perform truncation, high-pass, and exponential filtering on an Fid

     \param f Fid to filter
     \return Fid Filtered Fid
    */
    Fid filterFid(const Fid f);

    void setStart(double s) { d_start = s; }
    void setEnd(double e) { d_end = e; }
    void setPzf(int z) { d_pzf = z; }
    void setScaling(double s) { d_scaling = s; }
    void setIgnoreZone(double z) { d_ignoreZone = z; }

private:
    gsl_fft_real_wavetable *real; /*!< Wavetable for GNU Scientific Library FFT operations */
    gsl_fft_real_workspace *work; /*!< Memory for GNU Scientific Library FFT operations */
    int d_numPnts; /*!< Number of points used to allocate last wavetable and workspace */

    double d_start;
    double d_end;
    int d_pzf;
    double d_scaling;
    double d_ignoreZone;

    QVector<qint64> d_fidData;

};

#endif // FTWORKER_H

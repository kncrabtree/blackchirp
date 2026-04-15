#ifndef MAINFTPLOT_H
#define MAINFTPLOT_H

#include "ftplot.h"
#include <QObject>
#include <memory>

namespace BC::Key::FtMainPlot {
inline constexpr QLatin1StringView id{"Main"};
}

class MainFtPlot : public FtPlot
{
    Q_OBJECT
public:
    MainFtPlot(QWidget *parent = nullptr);
    ~MainFtPlot();
    
    virtual void prepareForExperiment(const Experiment &e);
    
public slots:
    void newPeakList(const QVector<QPointF> l);
    
private:
    std::unique_ptr<BlackchirpPlotCurve> p_peakData;
    
};

#endif // MAINFTPLOT_H

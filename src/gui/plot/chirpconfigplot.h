#ifndef CHIRPCONFIGPLOT_H
#define CHIRPCONFIGPLOT_H

#include <gui/plot/zoompanplot.h>
#include <memory>
#include <vector>

class ChirpConfig;
class BlackchirpPlotCurve;

namespace BC::Key {
static const QString chirpPlot{"ChirpConfigPlot"};
static const QString chirpCurve{"Chirp"};
}

class ChirpConfigPlot : public ZoomPanPlot
{
public:
    ChirpConfigPlot(QWidget *parent =0);
    ~ChirpConfigPlot();

public slots:
    void newChirp(const ChirpConfig cc);

private:
    std::unique_ptr<BlackchirpPlotCurve> p_chirpCurve;
    std::vector<std::unique_ptr<BlackchirpPlotCurve>> d_markerCurves;
};

#endif // CHIRPCONFIGPLOT_H

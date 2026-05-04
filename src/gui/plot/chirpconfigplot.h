#ifndef CHIRPCONFIGPLOT_H
#define CHIRPCONFIGPLOT_H

#include <gui/plot/zoompanplot.h>
#include <memory>

class ChirpConfig;
class BlackchirpPlotCurve;

namespace BC::Key {
static const QString chirpPlot{"ChirpConfigPlot"};
static const QString chirpCurve{"Chirp"};
static const QString ampCurve{"AmpEnable"};
static const QString protCurve{"Protection"};
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
    std::unique_ptr<BlackchirpPlotCurve> p_ampEnableCurve;
    std::unique_ptr<BlackchirpPlotCurve> p_protectionCurve;
};

#endif // CHIRPCONFIGPLOT_H

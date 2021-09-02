#ifndef CHIRPCONFIGPLOT_H
#define CHIRPCONFIGPLOT_H

#include <gui/plot/zoompanplot.h>

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
    BlackchirpPlotCurve *p_ampEnableCurve, *p_protectionCurve, *p_chirpCurve;
};

#endif // CHIRPCONFIGPLOT_H

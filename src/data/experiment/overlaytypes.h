#ifndef OVERLAYTYPES_H
#define OVERLAYTYPES_H

#include <data/experiment/overlaybase.h>
#include <data/analysis/ft.h>

namespace BC::Key::Overlay {
// FT metadata keys
static const QString ftYMin{"ftYMin"};
static const QString ftYMax{"ftYMax"};
static const QString ftX0MHz{"ftX0MHz"};
static const QString ftSpacingMHz{"ftSpacingMHz"};
static const QString ftLoFreqMHz{"ftLoFreqMHz"};
static const QString ftShots{"ftShots"};

// Catalog overlay specific keys
namespace Catalog {
    static const QString sourceProgram{"catalogSourceProgram"};
    static const QString moleculeName{"catalogMoleculeName"};
    static const QString convolutionEnabled{"catalogConvolutionEnabled"};
    static const QString lineshapeType{"catalogLineshapeType"};
    static const QString linewidth{"catalogLinewidth"};
    static const QString convolutionMinFreq{"catalogConvolutionMinFreq"};
    static const QString convolutionMaxFreq{"catalogConvolutionMaxFreq"};
    static const QString pointSpacing{"catalogPointSpacing"};
    static const QString transitionCount{"catalogTransitionCount"};
    static const QString frequencyRange{"catalogFrequencyRange"};
}
}

class BCExpOverlay : public OverlayBase
{
public:
    BCExpOverlay();

    // OverlayBase interface
private:
    QVector<QPointF> _xyData() const override;

public:

    // Set FT data directly (called after creation)
    void setFtData(const Ft &ftData);

protected:
    void readFromDest() override;
    void writeToDest() override;
    void _storeMetadata(std::map<QString, QVariant> &m) override;
    void _retrieveMetadata(const std::map<QString, QVariant> &m) override;


private:
    Ft d_ft;

};

#endif // OVERLAYTYPES_H


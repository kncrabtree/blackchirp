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
}

class BCExpOverlay : public OverlayBase
{
public:
    BCExpOverlay();

    // OverlayBase interface
    QVector<QPointF> xyData() const override;

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


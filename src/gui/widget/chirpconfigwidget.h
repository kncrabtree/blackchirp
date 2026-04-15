#ifndef CHIRPCONFIGWIDGET_H
#define CHIRPCONFIGWIDGET_H

#include <QWidget>

#include <data/storage/settingsstorage.h>
#include <data/experiment/rfconfig.h>
#include <data/model/chirptablemodel.h>
#include <data/model/markertablemodel.h>

#include <memory>

class QSpinBox;

namespace Ui {
class ChirpConfigWidget;
}

namespace BC::Key::ChirpConfigWidget {
inline constexpr QLatin1StringView key{"ChirpConfigWidget"};
inline constexpr QLatin1StringView numChirps{"numChirps"};
inline constexpr QLatin1StringView interval{"chirpIntervalUs"};
inline constexpr QLatin1StringView applyAll{"applyToAll"};
}

/*!
 * \brief The ChirpConfigWidget class
 *
 * The post chirp gate and post chirp protection are both measured with respect to the end of the chirp!
 *
 */
class ChirpConfigWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit ChirpConfigWidget(QWidget *parent = 0);
    ~ChirpConfigWidget();

    void initialize(const RfConfig &rfc);
    void setFromRfConfig(const RfConfig &rfc);
    ChirpConfig &getChirps();

public slots:
    void enableEditing(bool enabled);
    void setButtonStates();

    void addSegment();
    void addEmptySegment();
    void insertSegment();
    void insertEmptySegment();
    void moveSegments(int direction);
    void removeSegments();
    void clear();

    void updateChirpPlot();

signals:
    void chirpConfigChanged();


private:
    void updateRfConfig();
    Ui::ChirpConfigWidget *ui;
    ChirpTableModel *p_ctm;
    MarkerTableModel *p_mtm;
    bool d_rampOnly;
    double d_awgSampleRate;
    RfConfig d_rfConfig;

    bool isSelectionContiguous(QModelIndexList l);
    void clearList(bool replot=true);



};

#endif // CHIRPCONFIGWIDGET_H

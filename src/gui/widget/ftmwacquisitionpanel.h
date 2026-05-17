#ifndef FTMWACQUISITIONPANEL_H
#define FTMWACQUISITIONPANEL_H

#include <QWidget>

class QSpinBox;
class QPushButton;

class FtmwAcquisitionPanel : public QWidget
{
    Q_OBJECT
public:
    explicit FtmwAcquisitionPanel(bool main, QWidget *parent = nullptr);

    int refreshInterval() const;

    void setRefreshInterval(int ms);
    void setRefreshEnabled(bool enabled);

    void setAverages(int n);
    void setPeakUpControlsEnabled(bool enabled);

    void setManualBackupEnabled(bool enabled);

signals:
    void refreshIntervalChanged(int ms);
    void averagesChanged(int n);
    void resetAveragesClicked();
    void manualBackupClicked();

private:
    QSpinBox *p_refreshBox;
    QSpinBox *p_averagesBox;
    QPushButton *p_resetAveragesButton;
    QPushButton *p_manualBackupButton;
    bool d_main;
};

#endif // FTMWACQUISITIONPANEL_H

#ifndef LIFPROCESSINGWIDGET_H
#define LIFPROCESSINGWIDGET_H

#include <QWidget>

#include <data/storage/settingsstorage.h>

namespace BC::Key::LifProcessing {
const QString key("lifProcessingWidget");
const QString lgStart("lifGateStart");
const QString lgEnd("lifGateEnd");
const QString rgStart("refGateStart");
const QString rgEnd("refGateEnd");
const QString lpAlpha("lowPassAlpha");
const QString sgEn("savGolEnabled");
const QString sgWin("savGolWindow");
const QString sgPoly("savGolPoly");
}

class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QPushButton;

class LifProcessingWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit LifProcessingWidget(bool store = true, QWidget *parent = nullptr);

    struct LifProcSettings {
        int lifGateStart{-1};
        int lifGateEnd{-1};
        int refGateStart{-1};
        int refGateEnd{-1};
        double lowPassAlpha{};
        bool savGolEnabled{false};
        int savGolWin{11};
        int savGolPoly{3};
    };

    void initialize(int recLen, bool ref);
    void setAll(const LifProcSettings &lc);
    LifProcSettings getSettings() const;

signals:
    void lgStartChanged(int);
    void lgEndChanged(int);
    void rgStartChanged(int);
    void rgEndChanged(int);

private:
    QSpinBox *p_lgStartBox, *p_lgEndBox, *p_rgStartBox, *p_rgEndBox, *p_sgWinBox, *p_sgPolyBox;
    QDoubleSpinBox *p_lpAlphaBox;
    QCheckBox *p_sgEnBox;
    QPushButton *p_reprocessButton;
};

#endif // LIFPROCESSINGWIDGET_H

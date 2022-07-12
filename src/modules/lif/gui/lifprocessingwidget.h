#ifndef LIFPROCESSINGWIDGET_H
#define LIFPROCESSINGWIDGET_H

#include <QWidget>

#include <data/storage/settingsstorage.h>
#include <modules/lif/data/liftrace.h>

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

    void initialize(int recLen, bool ref);
    void setAll(const LifTrace::LifProcSettings &lc);
    LifTrace::LifProcSettings getSettings() const;

    void experimentComplete();

signals:
    void settingChanged();

private:
    QSpinBox *p_lgStartBox, *p_lgEndBox, *p_rgStartBox, *p_rgEndBox, *p_sgWinBox, *p_sgPolyBox;
    QDoubleSpinBox *p_lpAlphaBox;
    QCheckBox *p_sgEnBox;
    QPushButton *p_reprocessButton;
};

#endif // LIFPROCESSINGWIDGET_H

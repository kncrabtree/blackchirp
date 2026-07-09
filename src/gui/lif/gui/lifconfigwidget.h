#ifndef LIFCONFIGWIDGET_H
#define LIFCONFIGWIDGET_H

#include <QWidget>

#include <data/loadout/hardwareloadout.h>

class QTabWidget;
class LifControlWidget;
class LifConversionWidget;
class LifConfig;

namespace BC::Key::LifConfigWidget {
inline constexpr QLatin1StringView key{"LifConfigWidget"};
}

/*!
 * \brief Tabbed Acquisition + Conversion composite, the LIF analog of
 *        FtmwConfigWidget.
 *
 * Hosts a LifControlWidget ("Acquisition") and a LifConversionWidget
 * ("Conversion") as tabs. Reused by both ExperimentLifConfigPage (wizard
 * page) and LifConfigDialog (standalone "LIF Configuration" dialog) —
 * mirrors how FtmwConfigWidget is shared by ExperimentFtmwConfigPage and
 * FtmwConfigDialog. The trailing bool is forwarded to the conversion tab's
 * showDeleteButton, exactly as FtmwConfigWidget's does for its own preset
 * bar: the page passes the page-variant (hidden), the dialog passes the
 * dialog-variant (shown).
 *
 * Only the conversion tab carries preset/dirty state — LifControlWidget's
 * acquisition settings are device-local hardware settings, not part of the
 * per-experiment/preset system — so isDirty()/clearDirty()/toLifPreset()
 * simply delegate to the conversion widget.
 */
class LifConfigWidget : public QWidget
{
    Q_OBJECT
public:
    explicit LifConfigWidget(const QString &digitizerHwKey, const QString &laserHwKey,
                              bool showDeleteButton = true, QWidget *parent = nullptr);

    LifControlWidget *lifControlWidget() const { return p_lcw; }
    LifConversionWidget *lifConversionWidget() const { return p_conversionWidget; }

    void setFromConfig(const LifConfig &cfg);
    void toConfig(LifConfig &cfg);
    //! Snapshot the conversion tab's current wiring into a LifPreset (mirrors LifConversionWidget::toLifPreset()).
    LifPreset toLifPreset() const;

    bool isDirty() const;

signals:
    //! Re-exposed from LifConversionWidget: fired on any wiring/FINAL/harmonic edit.
    void edited();
    void dirtyChanged(bool dirty);

public slots:
    void clearDirty();

private:
    QTabWidget *p_tabWidget;
    LifControlWidget *p_lcw;
    LifConversionWidget *p_conversionWidget;
};

#endif // LIFCONFIGWIDGET_H

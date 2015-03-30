#ifndef FTMWCONFIGWIDGET_H
#define FTMWCONFIGWIDGET_H

#include <QWidget>
#include "ftmwconfig.h"

class QComboBox;

//Index definitions for convenience
#define FCW_MODETARGETSHOTS 0
#define FCW_MODETARGETTIME 1
#define FCW_MODEFOREVER 2
#define FCW_MODEPEAKUP 3

#define FCW_2GSS 0
#define FCW_5GSS 1
#define FCW_10GSS 2
#define FCW_20GSS 3
#define FCW_50GSS 4
#define FCW_100GSS 5

#define FCW_UPPERSIDEBAND 0
#define FCW_LOWERSIDEBAND 1

namespace Ui {
class FtmwConfigWidget;
}

class FtmwConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit FtmwConfigWidget(QWidget *parent = 0);
    ~FtmwConfigWidget();

    void setFromConfig(const FtmwConfig config);
    FtmwConfig getConfig() const;

public slots:
    void loadFromSettings();
    void saveToSettings();
    void configureUI();
    void validateSpinboxes();

private:
    Ui::FtmwConfigWidget *ui;

    void setComboBoxIndex(QComboBox *box, QVariant value);

};

#endif // FTMWCONFIGWIDGET_H

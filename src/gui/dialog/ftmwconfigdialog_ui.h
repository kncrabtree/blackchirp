#ifndef FTMWCONFIGDIALOG_UI_H
#define FTMWCONFIGDIALOG_UI_H

#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

#include <gui/widget/rfconfigwidget.h>
#include <gui/widget/chirpconfigwidget.h>
#include <gui/widget/ftmwdigitizerconfigwidget.h>

class Ui_FtmwConfigDialog
{
public:
    QVBoxLayout *mainLayout;
    QTabWidget *tabWidget;

    QWidget *rfTab;
    QVBoxLayout *rfTabLayout;
    QHBoxLayout *rfSourceRow;
    QLabel *rfSourceLabel;
    QComboBox *rfSourceCombo;
    RfConfigWidget *rfWidget;

    QWidget *chirpTab;
    QVBoxLayout *chirpTabLayout;
    QHBoxLayout *chirpSourceRow;
    QLabel *chirpSourceLabel;
    QComboBox *chirpSourceCombo;
    ChirpConfigWidget *chirpWidget;

    QWidget *digiTab;
    QVBoxLayout *digiTabLayout;
    QHBoxLayout *digiSourceRow;
    QLabel *digiSourceLabel;
    QComboBox *digiSourceCombo;
    FtmwDigitizerConfigWidget *digiWidget;

    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *dialog)
    {
        dialog->setWindowTitle("FTMW Configuration");
        dialog->resize(900, 650);

        mainLayout = new QVBoxLayout(dialog);

        tabWidget = new QTabWidget(dialog);

        // RF Config tab
        rfTab = new QWidget;
        rfTabLayout = new QVBoxLayout(rfTab);
        rfSourceRow = new QHBoxLayout;
        rfSourceLabel = new QLabel("Load from loadout:");
        rfSourceCombo = new QComboBox;
        rfSourceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        rfSourceRow->addWidget(rfSourceLabel);
        rfSourceRow->addWidget(rfSourceCombo, 1);
        rfTabLayout->addLayout(rfSourceRow);
        rfWidget = new RfConfigWidget(rfTab);
        rfTabLayout->addWidget(rfWidget, 1);
        tabWidget->addTab(rfTab, "RF Config");

        // Chirp Config tab
        chirpTab = new QWidget;
        chirpTabLayout = new QVBoxLayout(chirpTab);
        chirpSourceRow = new QHBoxLayout;
        chirpSourceLabel = new QLabel("Load from loadout:");
        chirpSourceCombo = new QComboBox;
        chirpSourceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        chirpSourceRow->addWidget(chirpSourceLabel);
        chirpSourceRow->addWidget(chirpSourceCombo, 1);
        chirpTabLayout->addLayout(chirpSourceRow);
        chirpWidget = new ChirpConfigWidget(chirpTab);
        chirpTabLayout->addWidget(chirpWidget, 1);
        tabWidget->addTab(chirpTab, "Chirp Config");

        // Digitizer Config tab
        digiTab = new QWidget;
        digiTabLayout = new QVBoxLayout(digiTab);
        digiSourceRow = new QHBoxLayout;
        digiSourceLabel = new QLabel("Load from loadout:");
        digiSourceCombo = new QComboBox;
        digiSourceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        digiSourceRow->addWidget(digiSourceLabel);
        digiSourceRow->addWidget(digiSourceCombo, 1);
        digiTabLayout->addLayout(digiSourceRow);
        digiWidget = new FtmwDigitizerConfigWidget(digiTab);
        digiTabLayout->addWidget(digiWidget, 1);
        tabWidget->addTab(digiTab, "Digitizer Config");

        mainLayout->addWidget(tabWidget, 1);

        buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, dialog);
        mainLayout->addWidget(buttonBox);
    }
};

namespace Ui {
class FtmwConfigDialog : public Ui_FtmwConfigDialog {};
}

#endif // FTMWCONFIGDIALOG_UI_H

#ifndef BATCHSEQUENCEDIALOG_H
#define BATCHSEQUENCEDIALOG_H

#include <QDialog>
#include <data/storage/settingsstorage.h>

namespace Ui {
class BatchSequenceDialog;
}

namespace BC::Key::SeqDialog {
static const QString key("BatchSequenceDialog");
static const QString batchExperiments("numExpts");
static const QString batchInterval("interval");
}

class BatchSequenceDialog : public QDialog, public SettingsStorage
{
    Q_OBJECT

public:
    explicit BatchSequenceDialog(QWidget *parent = 0);
    ~BatchSequenceDialog();

    void setQuickExptEnabled(bool en);

    int numExperiments() const;
    int interval() const;

    static const int configureCode = 23;
    static const int quickCode = 27;

private:
    Ui::BatchSequenceDialog *ui;


};

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

class Ui_BatchSequenceDialog
{
public:
    QVBoxLayout *verticalLayout;
    QLabel *label;
    QWidget *widget;
    QFormLayout *formLayout;
    QLabel *numberOfExperimentsLabel;
    QSpinBox *numberOfExperimentsSpinBox;
    QLabel *timeBetweenExperimentsLabel;
    QSpinBox *timeBetweenExperimentsSpinBox;
    QHBoxLayout *horizontalLayout;
    QPushButton *cancelButton;
    QSpacerItem *horizontalSpacer;
    QPushButton *quickButton;
    QPushButton *configureButton;

    void setupUi(QDialog *BatchSequenceDialog)
    {
        if (BatchSequenceDialog->objectName().isEmpty())
            BatchSequenceDialog->setObjectName(QString::fromUtf8("BatchSequenceDialog"));
        BatchSequenceDialog->resize(416, 183);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/bc.png"), QSize(), QIcon::Normal, QIcon::Off);
        BatchSequenceDialog->setWindowIcon(icon);
        verticalLayout = new QVBoxLayout(BatchSequenceDialog);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        label = new QLabel(BatchSequenceDialog);
        label->setObjectName(QString::fromUtf8("label"));
        label->setWordWrap(true);

        verticalLayout->addWidget(label);

        widget = new QWidget(BatchSequenceDialog);
        widget->setObjectName(QString::fromUtf8("widget"));
        formLayout = new QFormLayout(widget);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        numberOfExperimentsLabel = new QLabel(widget);
        numberOfExperimentsLabel->setObjectName(QString::fromUtf8("numberOfExperimentsLabel"));

        formLayout->setWidget(0, QFormLayout::LabelRole, numberOfExperimentsLabel);

        numberOfExperimentsSpinBox = new QSpinBox(widget);
        numberOfExperimentsSpinBox->setObjectName(QString::fromUtf8("numberOfExperimentsSpinBox"));
        numberOfExperimentsSpinBox->setMinimum(1);
        numberOfExperimentsSpinBox->setMaximum(1000);

        formLayout->setWidget(0, QFormLayout::FieldRole, numberOfExperimentsSpinBox);

        timeBetweenExperimentsLabel = new QLabel(widget);
        timeBetweenExperimentsLabel->setObjectName(QString::fromUtf8("timeBetweenExperimentsLabel"));

        formLayout->setWidget(1, QFormLayout::LabelRole, timeBetweenExperimentsLabel);

        timeBetweenExperimentsSpinBox = new QSpinBox(widget);
        timeBetweenExperimentsSpinBox->setObjectName(QString::fromUtf8("timeBetweenExperimentsSpinBox"));
        timeBetweenExperimentsSpinBox->setMaximum(1000000000);
        timeBetweenExperimentsSpinBox->setSingleStep(60);
        timeBetweenExperimentsSpinBox->setValue(300);

        formLayout->setWidget(1, QFormLayout::FieldRole, timeBetweenExperimentsSpinBox);


        verticalLayout->addWidget(widget);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        cancelButton = new QPushButton(BatchSequenceDialog);
        cancelButton->setObjectName(QString::fromUtf8("cancelButton"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/abort.png"), QSize(), QIcon::Normal, QIcon::Off);
        cancelButton->setIcon(icon1);
        cancelButton->setIconSize(QSize(15, 15));

        horizontalLayout->addWidget(cancelButton);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        quickButton = new QPushButton(BatchSequenceDialog);
        quickButton->setObjectName(QString::fromUtf8("quickButton"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/quickexpt.png"), QSize(), QIcon::Normal, QIcon::Off);
        quickButton->setIcon(icon2);
        quickButton->setIconSize(QSize(15, 15));

        horizontalLayout->addWidget(quickButton);

        configureButton = new QPushButton(BatchSequenceDialog);
        configureButton->setObjectName(QString::fromUtf8("configureButton"));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/icons/configure.png"), QSize(), QIcon::Normal, QIcon::Off);
        configureButton->setIcon(icon3);
        configureButton->setIconSize(QSize(15, 15));

        horizontalLayout->addWidget(configureButton);


        verticalLayout->addLayout(horizontalLayout);


        retranslateUi(BatchSequenceDialog);

        QMetaObject::connectSlotsByName(BatchSequenceDialog);
    } // setupUi

    void retranslateUi(QDialog *BatchSequenceDialog)
    {
        BatchSequenceDialog->setWindowTitle(QApplication::translate("BatchSequenceDialog", "Experiment Sequence Setup", nullptr));
        label->setText(QApplication::translate("BatchSequenceDialog", "In this mode, the program will perform a sequence of identical experiments. The time between experiments is measured from the end of one experiment to the start of the next.", nullptr));
        numberOfExperimentsLabel->setText(QApplication::translate("BatchSequenceDialog", "Number of Experiments", nullptr));
        timeBetweenExperimentsLabel->setText(QApplication::translate("BatchSequenceDialog", "Time between Experiments", nullptr));
        timeBetweenExperimentsSpinBox->setSuffix(QApplication::translate("BatchSequenceDialog", " s", nullptr));
        cancelButton->setText(QApplication::translate("BatchSequenceDialog", "Cancel", nullptr));
        quickButton->setText(QApplication::translate("BatchSequenceDialog", "Quick Experiment", nullptr));
        configureButton->setText(QApplication::translate("BatchSequenceDialog", "Configure Experiment", nullptr));
    } // retranslateUi

};

namespace Ui {
    class BatchSequenceDialog: public Ui_BatchSequenceDialog {};
} // namespace Ui

#endif // BATCHSEQUENCEDIALOG_H

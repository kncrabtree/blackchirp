#include <gui/dialog/quickexptdialog.h>
#include "ui_quickexptdialog.h"

QuickExptDialog::QuickExptDialog(std::shared_ptr<Experiment> e, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::QuickExptDialog)
{
    ui->setupUi(this);

    connect(ui->cancelButton,&QPushButton::clicked,this,&QuickExptDialog::reject);
    connect(ui->configureButton,&QPushButton::clicked,this,[=](){ done(d_configureResult); });
    connect(ui->startButton,&QPushButton::clicked,this,&QuickExptDialog::accept);

    QString html;

    //generate summary text and insert header details into table widget
    if(e->ftmwConfig().isEnabled())
    {
        html.append(QString("<h1>FTMW settings</h1>"));
        html.append(QString("<ul>"));
        if(e->ftmwConfig().type() == BlackChirp::FtmwTargetShots)
        {
            html.append(QString("<li>Mode: Target Shots</li>"));
            html.append(QString("<li>Shots: %1</li>").arg(e->ftmwConfig().targetShots()));
        }
        else if(e->ftmwConfig().type() == BlackChirp::FtmwPeakUp)
        {
            html.append(QString("<li>Mode: Peak Up</li>"));
            html.append(QString("<li>Shots: %1</li>").arg(e->ftmwConfig().targetShots()));
        }
        else if(e->ftmwConfig().type() == BlackChirp::FtmwTargetTime)
        {
            html.append(QString("<li>Mode: Target Time</li>"));
            html.append(QString("<li>End time: %1</li>").arg(e->ftmwConfig().targetTime().toString()));
        }
        else if(e->ftmwConfig().type() == BlackChirp::FtmwForever)
        {
            html.append(QString("<li>Mode: Forever</li>"));
        }
        html.append(QString("<li>Chirps: %1</li>").arg(e->ftmwConfig().chirpConfig().numChirps()));
        html.append(QString("<li>Sample rate: %1 GS/s</li>").arg(e->ftmwConfig().scopeConfig().d_sampleRate/1e9,0,'f',0));
        html.append(QString("<li>Record length: %1</li>").arg(e->ftmwConfig().scopeConfig().d_recordLength));
        if(e->ftmwConfig().chirpConfig().numChirps() > 1)
        {
            html.append(QString("<li>Chirp spacing: %1 &mu;s</li>").arg(e->ftmwConfig().chirpConfig().chirpInterval(),0,'f',1));
        }
        html.append(QString("</ul>"));
    }
    else
    {
        html.append(QString("<h1>FTMW disabled</h1>"));
    }

#ifdef BC_LIF
    if(e.lifConfig().isEnabled())
    {
        html.append(QString("<h1>LIF settings</h1><ul>"));
        html.append(QString("<li>Delay range: %1-%2 &mu;s</li>").arg(e.lifConfig().delayRange().first,0,'f',3).arg(e.lifConfig().delayRange().second,0,'f',3));
        html.append(QString("<li>Delay step: %1 &mu;s</li>").arg(e.lifConfig().delayStep(),0,'f',3));
        html.append(QString("<li>Frequency range: %1-%2 1/cm</li>").arg(e.lifConfig().laserRange().first,0,'f',3).arg(e.lifConfig().laserRange().second,0,'f',3));
        html.append(QString("<li>Frequency step: %1 1/cm</li>").arg(e.lifConfig().laserStep(),0,'f',3));
        html.append(QString("<li>Shots per point: %1</li>").arg(e.lifConfig().shotsPerPoint()));
        html.append(QString("<li>Total shots: %1</li>").arg(e.lifConfig().totalShots()));
        html.append(QString("</ul>"));
    }
    else
    {
        html.append(QString("<h1>LIF disabled</h1>"));
    }
#endif

#ifdef BC_MOTOR
    if(e.motorScan().isEnabled())
    {
        html.append(QString("<h1>Motor Scan Settings</h1><ul>"));
        html.append(QString("<li>X Range: %1-%2 mm</li>")
                    .arg(e.motorScan().xVal(0)).arg(e.motorScan().xVal(e.motorScan().xPoints()-1)));
        html.append(QString("<li>X Points: %1</li>").arg(e.motorScan().xPoints()));
        html.append(QString("<li>Y Range: %1-%2 mm</li>")
                    .arg(e.motorScan().yVal(0)).arg(e.motorScan().yVal(e.motorScan().yPoints()-1)));
        html.append(QString("<li>Y Points: %1</li>").arg(e.motorScan().yPoints()));
        html.append(QString("<li>Z Range: %1-%2 mm</li>")
                    .arg(e.motorScan().zVal(0)).arg(e.motorScan().zVal(e.motorScan().zPoints()-1)));
        html.append(QString("<li>Z Points: %1</li>").arg(e.motorScan().zPoints()));
        html.append(QString("<li>Shots Per Point: %1</li>").arg(e.motorScan().shotsPerPoint()));
        html.append(QString("</ul>"));
    }
#endif

    ui->textEdit->insertHtml(html);

    ui->tableWidget->setHorizontalHeaderItem(0,new QTableWidgetItem(QString("Key")));
    ui->tableWidget->setHorizontalHeaderItem(1,new QTableWidgetItem(QString("Value")));
    ui->tableWidget->setHorizontalHeaderItem(2,new QTableWidgetItem(QString("Unit")));
    ui->tableWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->tableWidget->setColumnCount(3);
    ui->tableWidget->setEditTriggers(QTableWidget::NoEditTriggers);
    ui->tableWidget->setSelectionBehavior(QAbstractItemView::SelectRows);

#pragma message("Update experiment summary")
    auto header = e->headerMap();
    auto it = header.constBegin();
    ui->tableWidget->setRowCount(header.size());
    int i = 0;
    while(it != header.constEnd())
    {
        ui->tableWidget->setItem(i,0,new QTableWidgetItem(it.key()));
        ui->tableWidget->setItem(i,1,new QTableWidgetItem(it.value().first.toString()));
        ui->tableWidget->setItem(i,2,new QTableWidgetItem(it.value().second));

        it++;
        i++;
    }
    ui->tableWidget->resizeColumnsToContents();
}

QuickExptDialog::~QuickExptDialog()
{
    delete ui;
}

#include "quickexptdialog.h"
#include "ui_quickexptdialog.h"

QuickExptDialog::QuickExptDialog(Experiment e, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::QuickExptDialog)
{
    ui->setupUi(this);

    connect(ui->cancelButton,&QPushButton::clicked,this,&QuickExptDialog::reject);
    connect(ui->configureButton,&QPushButton::clicked,this,[=](){ done(d_configureResult); });
    connect(ui->startButton,&QPushButton::clicked,this,&QuickExptDialog::accept);

    QString html;

    //generate summary text and insert header details into table widget
    if(e.ftmwConfig().isEnabled())
    {
        html.append(QString("<h1>FTMW settings</h1>"));
        html.append(QString("<ul>"));
        if(e.ftmwConfig().type() == BlackChirp::FtmwTargetShots)
        {
            html.append(QString("<li>Mode: Target Shots</li>"));
            html.append(QString("<li>Shots: %1</li>").arg(e.ftmwConfig().targetShots()));
        }
        else if(e.ftmwConfig().type() == BlackChirp::FtmwPeakUp)
        {
            html.append(QString("<li>Mode: Peak Up</li>"));
            html.append(QString("<li>Shots: %1</li>").arg(e.ftmwConfig().targetShots()));
        }
        else if(e.ftmwConfig().type() == BlackChirp::FtmwTargetTime)
        {
            html.append(QString("<li>Mode: Target Time</li>"));
            html.append(QString("<li>End time: %1</li>").arg(e.ftmwConfig().targetTime().toString()));
        }
        else if(e.ftmwConfig().type() == BlackChirp::FtmwForever)
        {
            html.append(QString("<li>Mode: Forever</li>"));
        }
        html.append(QString("<li>Chirps: %1</li>").arg(e.ftmwConfig().chirpConfig().numChirps()));
        html.append(QString("<li>Sample rate: %1 GS/s</li>").arg(e.ftmwConfig().scopeConfig().sampleRate/1e9,0,'f',0));
        html.append(QString("<li>Record length: %1</li>").arg(e.ftmwConfig().scopeConfig().recordLength));
        if(e.ftmwConfig().chirpConfig().numChirps() > 1)
        {
            html.append(QString("<li>Chirp spacing: %1 &mu;s</li>").arg(e.ftmwConfig().chirpConfig().chirpInterval(),0,'f',1));
            if(e.ftmwConfig().scopeConfig().summaryFrame)
                html.append(QString("<li>Summary frame: Yes</li>"));
            else
                html.append(QString("<li>Summary frame: No</li>"));
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
        html.append(QString("<li>Frequency range: %1-%2 1/cm</li>").arg(e.lifConfig().frequencyRange().first,0,'f',3).arg(e.lifConfig().frequencyRange().second,0,'f',3));
        html.append(QString("<li>Frequency step: %1 1/cm</li>").arg(e.lifConfig().frequencyStep(),0,'f',3));
        html.append(QString("<li>Shots per point: %1</li>").arg(e.lifConfig().shotsPerPoint()));
        html.append(QString("<li>Total shots: %1</li>").arg(e.lifConfig().totalShots()));
        html.append(QString("</ul>"));
    }
    else
    {
        html.append(QString("<h1>LIF disabled</h1>"));
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

    auto header = e.headerMap();
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

bool QuickExptDialog::sleepWhenDone() const
{
    return ui->sleepCheckBox->isChecked();
}

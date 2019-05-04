namespace ProgressReporter
{
    partial class MainForm
    {
        private System.ComponentModel.IContainer components = null;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.CurrentTaskBox = new System.Windows.Forms.GroupBox();
            this.CurrentProgressBar = new System.Windows.Forms.ProgressBar();
            this.OverallTaskBox = new System.Windows.Forms.GroupBox();
            this.OverallProgressBar = new System.Windows.Forms.ProgressBar();
            this.MainTimer = new System.Windows.Forms.Timer(this.components);
            this.CurrentTaskBox.SuspendLayout();
            this.OverallTaskBox.SuspendLayout();
            this.SuspendLayout();
            // 
            // CurrentTaskBox
            // 
            this.CurrentTaskBox.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.CurrentTaskBox.Controls.Add(this.CurrentProgressBar);
            this.CurrentTaskBox.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.CurrentTaskBox.Location = new System.Drawing.Point(12, 13);
            this.CurrentTaskBox.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.CurrentTaskBox.Name = "CurrentTaskBox";
            this.CurrentTaskBox.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.CurrentTaskBox.Size = new System.Drawing.Size(823, 45);
            this.CurrentTaskBox.TabIndex = 0;
            this.CurrentTaskBox.TabStop = false;
            this.CurrentTaskBox.Text = "Current task 123/4567";
            // 
            // CurrentProgressBar
            // 
            this.CurrentProgressBar.Location = new System.Drawing.Point(7, 24);
            this.CurrentProgressBar.Margin = new System.Windows.Forms.Padding(4);
            this.CurrentProgressBar.Name = "CurrentProgressBar";
            this.CurrentProgressBar.Size = new System.Drawing.Size(809, 13);
            this.CurrentProgressBar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.CurrentProgressBar.TabIndex = 0;
            this.CurrentProgressBar.Value = 30;
            // 
            // OverallTaskBox
            // 
            this.OverallTaskBox.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.OverallTaskBox.Controls.Add(this.OverallProgressBar);
            this.OverallTaskBox.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.OverallTaskBox.Location = new System.Drawing.Point(12, 66);
            this.OverallTaskBox.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.OverallTaskBox.Name = "OverallTaskBox";
            this.OverallTaskBox.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.OverallTaskBox.Size = new System.Drawing.Size(823, 45);
            this.OverallTaskBox.TabIndex = 1;
            this.OverallTaskBox.TabStop = false;
            this.OverallTaskBox.Text = "Overall progress 123/4567";
            // 
            // OverallProgressBar
            // 
            this.OverallProgressBar.Location = new System.Drawing.Point(7, 24);
            this.OverallProgressBar.Margin = new System.Windows.Forms.Padding(4);
            this.OverallProgressBar.Name = "OverallProgressBar";
            this.OverallProgressBar.Size = new System.Drawing.Size(809, 13);
            this.OverallProgressBar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.OverallProgressBar.TabIndex = 0;
            this.OverallProgressBar.Value = 70;
            // 
            // MainTimer
            // 
            this.MainTimer.Enabled = true;
            this.MainTimer.Tick += new System.EventHandler(this.MainTimer_Tick);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.ClientSize = new System.Drawing.Size(847, 123);
            this.Controls.Add(this.OverallTaskBox);
            this.Controls.Add(this.CurrentTaskBox);
            this.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.MaximizeBox = false;
            this.Name = "MainForm";
            this.Text = "GPU Lightmass Progress";
            this.CurrentTaskBox.ResumeLayout(false);
            this.OverallTaskBox.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        private System.Windows.Forms.GroupBox CurrentTaskBox;
        private System.Windows.Forms.ProgressBar CurrentProgressBar;
        private System.Windows.Forms.GroupBox OverallTaskBox;
        private System.Windows.Forms.ProgressBar OverallProgressBar;
        private System.Windows.Forms.Timer MainTimer;
    }
}


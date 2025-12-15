import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../providers/job_provider.dart';
import '../../theme/app_theme.dart';
import '../../widgets/form_input.dart';
import '../../widgets/primary_button.dart';
import '../recruiter/recruiter_dashboard_page.dart';
import '../role_selection_page.dart';
import '../seeker/seeker_shell.dart';
import 'register_page.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key, this.initialRole = 'job_seeker'});
  final String initialRole;

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final emailCtrl = TextEditingController();
  final passwordCtrl = TextEditingController();
  late String selectedRole;

  @override
  void initState() {
    super.initState();
    selectedRole = widget.initialRole;
  }

  @override
  void dispose() {
    emailCtrl.dispose();
    passwordCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded),
          onPressed: () => Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const RoleSelectionPage())),
        ),
        title: const Text('Welcome back'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 12),
            Row(
              children: [
                ChoiceChip(
                  label: const Text('Job Seeker'),
                  selected: selectedRole == 'job_seeker',
                  onSelected: (_) => setState(() => selectedRole = 'job_seeker'),
                ),
                const SizedBox(width: 10),
                ChoiceChip(
                  label: const Text('Recruiter'),
                  selected: selectedRole == 'recruiter',
                  onSelected: (_) => setState(() => selectedRole = 'recruiter'),
                ),
              ],
            ),
            const SizedBox(height: 24),
            FormInput(label: 'Email', controller: emailCtrl, hint: 'alex@example.com', keyboardType: TextInputType.emailAddress),
            const SizedBox(height: 16),
            FormInput(label: 'Password', controller: passwordCtrl, hint: '••••••••', obscure: true),
            const SizedBox(height: 20),
            PrimaryButton(
              label: 'Sign In',
              loading: auth.isLoading,
              onPressed: () async {
                await auth.login(emailCtrl.text.trim(), passwordCtrl.text);
                if (auth.session != null) {
                  _goToDashboard(context, auth);
                  context.read<JobProvider>().loadJobs();
                } else if (auth.error != null) {
                  _showError(context, auth.error!);
                }
              },
            ),
            if (auth.error != null) ...[
              const SizedBox(height: 12),
              Text(auth.error!, style: const TextStyle(color: Colors.red, fontWeight: FontWeight.w700)),
            ],
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text('No account yet? '),
                TextButton(
                  onPressed: () {
                    Navigator.push(context, MaterialPageRoute(builder: (_) => RegisterPage(initialRole: selectedRole)));
                  },
                  child: const Text('Create one', style: TextStyle(color: AppTheme.primary, fontWeight: FontWeight.w700)),
                ),
              ],
            )
          ],
        ),
      ),
    );
  }

  void _goToDashboard(BuildContext context, AuthProvider auth) {
    final role = auth.session?.profile.role;
    if (role == 'recruiter') {
      Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (_) => const RecruiterDashboardPage()), (_) => false);
    } else {
      Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (_) => const SeekerShell()), (_) => false);
    }
  }

  void _showError(BuildContext context, String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }
}
